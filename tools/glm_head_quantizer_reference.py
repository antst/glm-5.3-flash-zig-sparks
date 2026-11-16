#!/usr/bin/python3.12
"""Owned GPU diagnostic of quantizer bytes versus the pinned Python formulas, not a model or speed gate."""
import ctypes as C
import ast
import dataclasses
import hashlib
import json
import pathlib
import struct
import sys

ROOT = pathlib.Path('/opt/glm')


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Driver:
    def __init__(self):
        self.lib = C.CDLL('libcuda.so.1')
        self.context = C.c_void_p()
        self.lib.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
        self.check(self.lib.cuCtxGetCurrent(C.byref(self.context)))
        assert self.context.value
        self.lib.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuModuleUnload.argtypes = [C.c_void_p]
        self.lib.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint]*7 + [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        self.modules = []

    @staticmethod
    def check(value):
        assert value == 0, ('CUDA Driver result', value)

    def function(self, path, name):
        data = path.read_bytes()
        storage = C.create_string_buffer(data)
        assert C.addressof(storage) % 8 == 0
        module, function = C.c_void_p(), C.c_void_p()
        self.check(self.lib.cuModuleLoadData(C.byref(module), storage))
        self.modules.append(module)
        self.check(self.lib.cuModuleGetFunction(C.byref(function), module, name.encode()))
        return function

    def launch(self, function, blocks, stream, values):
        args = (C.c_void_p*len(values))(*[C.cast(C.byref(x), C.c_void_p) for x in values])
        self.check(self.lib.cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, stream, args, None))

    def close(self):
        now = C.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(C.byref(now)))
        assert now.value == self.context.value
        for module in reversed(self.modules):
            self.check(self.lib.cuModuleUnload(module))


def reference_nodes(text):
    names = {'F8', 'make_f8', 'quantize4_mse_raw'}
    nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names and len(nodes) == 3
    return ast.Module(body=nodes, type_ignores=[])


def references(torch, path):
    # Execute only the pinned pure quantizer definitions, never import the old model or Triton module.
    scope = dict(torch=torch, dataclass=dataclasses.dataclass, B16=type('UnusedB16', (), {}),
                 F8_BLOCK=128, F8_MAX=448.0)
    exec(compile(reference_nodes(path.read_text()), str(path), 'exec'), scope)
    return scope


def mismatch(torch, got, want):
    a, b = got.view(torch.uint8).flatten(), want.view(torch.uint8).flatten()
    neq = a != b
    indexes = torch.nonzero(neq).flatten().cpu().tolist()
    return dict(bytes=a.numel(), different_bytes=len(indexes), first_offsets=indexes[:16],
                got_sha256=digest(bytes(a.cpu().tolist())), want_sha256=digest(bytes(b.cpu().tolist())))


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
    pins = json.loads((ROOT/'CAPTURE-INPUTS.json').read_text())
    for name, wanted in pins['files'].items():
        assert digest((ROOT/name).read_bytes()) == wanted, name
    import torch
    torch.cuda.set_device(0)
    generator = torch.Generator().manual_seed(7025)
    random = torch.randn((513, 128), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    random[:3].zero_()
    saved = json.loads((ROOT/'checkpoint-groups.json').read_text())
    actual = [g['bf16_words'] for t in saved['tensors'] for g in t['groups']]
    assert len(actual) == 224 and all(len(g) == 64 for g in actual)
    actual = torch.tensor(actual, dtype=torch.uint16).view(torch.bfloat16)
    src = torch.cat((random, actual.reshape(-1, 128))).contiguous().cuda()
    before = digest(bytes(src.view(torch.uint8).flatten().cpu().tolist()))
    ref = references(torch, ROOT/'python-qmm.py')
    reference = ref['make_f8'](src)
    scale, expected = reference.scale.flatten(), reference.weight.view(torch.uint8)
    g = src.float()
    got = torch.empty(src.shape, dtype=torch.uint8, device='cuda')
    got_s = torch.empty(scale.shape, dtype=torch.float32, device='cuda')
    status = torch.zeros(1, dtype=torch.int32, device='cuda')
    driver = Driver()
    completed = False
    try:
        fp8 = driver.function(ROOT/'glm_head_fp8.fatbin', 'tf_glm_head_fp8_quant_v1')
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        groups = src.shape[0]
        driver.launch(fp8, (groups+3)//4, stream, [C.c_uint64(src.data_ptr()), C.c_int64(groups),
            C.c_uint64(got.data_ptr()), C.c_uint64(got_s.data_ptr()), C.c_uint64(status.data_ptr())])
        torch.cuda.synchronize()
        assert status.item() == 0
        fp8_result = dict(codes=mismatch(torch, got, expected), scales=mismatch(torch, got_s, scale))
        maxima = g.abs().amax(-1).cpu().tolist()
        py_s = scale.cpu().tolist()
        fp8_result['scale_matches_CPU_RN_division'] = sum(a == max(f32(m/448.0), f32(1e-30)) for a, m in zip(py_s, maxima))
        fp8_result['scale_matches_CPU_reciprocal_multiply'] = sum(a == max(f32(m*f32(1.0/448.0)), f32(1e-30)) for a, m in zip(py_s, maxima))
        wm, sm, bm = ref['quantize4_mse_raw'](src)
        wm, sm, bm = wm.reshape(-1, 8), sm.flatten(), bm.flatten()
        nw, ns, nb = torch.empty_like(wm), torch.empty_like(sm), torch.empty_like(bm)
        status.zero_()
        mse = driver.function(ROOT/'glm_q4_mse.fatbin', 'tf_glm_q4_mse_v1')
        groups64 = src.numel()//64
        driver.launch(mse, (groups64+3)//4, stream, [C.c_uint64(src.data_ptr()), C.c_int64(groups64),
            C.c_uint64(nw.data_ptr()), C.c_uint64(ns.data_ptr()), C.c_uint64(nb.data_ptr()), C.c_uint64(status.data_ptr())])
        torch.cuda.synchronize()
        assert status.item() == 0
        mse_result = dict(words=mismatch(torch, nw, wm), scales=mismatch(torch, ns, sm), biases=mismatch(torch, nb, bm))
        dumps = {}
        tensors = dict(input=src, fp8_native=got, fp8_python=expected, scale_native=got_s, scale_python=scale,
                       mse_words_native=nw, mse_words_python=wm, mse_scale_native=ns, mse_scale_python=sm,
                       mse_bias_native=nb, mse_bias_python=bm)
        for name, tensor in tensors.items():
            data = bytes(tensor.contiguous().view(torch.uint8).flatten().cpu().tolist())
            pathlib.Path('/results', name+'.bin').write_bytes(data)
            dumps[name] = dict(bytes=len(data), sha256=digest(data))
        assert digest(bytes(src.view(torch.uint8).flatten().cpu().tolist())) == before
        torch.cuda.synchronize()
        completed = True
    finally:
        torch.cuda.synchronize()
        driver.close()
    result = dict(completed=completed, diagnostic=True, model_or_speed_claim=False, rank=int(sys.argv[4]),
        capability=list(torch.cuda.get_device_capability()), torch_version=torch.__version__, cuda=torch.version.cuda,
        source_commit=pins['source_commit'], source_manifest_sha256=pins['source_manifest_sha256'],
        input_sha256=before, fp8_groups=src.shape[0], mse_groups=groups64, dumps=dumps,
        fp8=fp8_result, mse=mse_result,
        exact_reference=all(v['different_bytes'] == 0 for x in (fp8_result, mse_result)
                            for v in x.values() if isinstance(v, dict)),
        scope='Quantizer byte observation only. 513 synthetic row128 groups plus 112 pairs of dense checkpoint '
              'groups; those pairs are not actual output-head groups. No model or speed qualification.')
    assert result['capability'] == [12, 1]
    pathlib.Path('/results/capture.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':main()
