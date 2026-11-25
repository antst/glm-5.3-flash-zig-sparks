# Building the native engine

Both topologies and all four EXL3 precision profiles use TensorFold native
1.3.0. The measured source commit is `26e17d87` and the tree is
`56bd5d3f53d435f6a16bc4db7229eb9df04ffaf7`. The signed archive snapshot preserves
every file in that tree. The measured executable SHA-256 is
`025ce1efce4a1d341b1b75d04e5bf0d998d25dfc7f56a98cb255e5b988c24907`.
[Signed source and manifest](SOURCE.json) · [Build inputs](BUILD-RECEIPTS.json)

## Native executable and CUDA assets

Use Zig 0.17.0 for Linux x86_64. The recorded compiler SHA-256 is
`7c61539af19fa4082c94848f1c2b57b89e76dc6cfd5bd4c57764c3caf8f82272`.
Extract `tensorfold-glm53-unified-source.tar.gz` and
`glm53-unified-embedded-images157.tar.gz` beside one another as `source/`
and `fatbins/`. From `source/`:

```sh
zig build native --prefix ../build-out -j1 --summary all \
  -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=safe \
  -Dversion=1.3.0 -Dstrip=true -Dfatbins=../fatbins -Dsm=121
```

The output is `build-out/native/bin/tensorfold-native`. Use the published
runtime archives to reproduce the measured executable identity. A rebuild must
be verified independently.

The optimized prefill modules load from the runtime archive. Their translation
units, local includes and per-module build information are supplied in
`glm53-unified-prefill-sources.tar.gz`. TP2 loads the 32-head module. The Q4
profiles load row-staged latent projection, while BF16 and Q8 use latent MMA.
[Release downloads](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.3.0-tp2-tp4)

## Retained NVFP4 executable

The optional retained NVFP4/BF16 mode has a separate byte-verified source snapshot and executable
SHA256 `312761ce390edf72d89d72645d6ccfd9a43d151e70f50ecb3f874f17d9a8ebfd`.
Its historical version is1.2.6. Extract its source into a separate directory and use `-Dversion=1.2.6`
with the same compiler/embedded-image inputs. Do not apply the coherent source's settings or cache
qualification to that older executable. The serving renderer keeps its native binary separate and
requires NVMe off. Its release source snapshot must also be explicitly signed with the human author.

## Structured-output library

The unified executable uses the measured `libtfgrammar.so`, SHA-256
`983f2323288bde6e6a6486984bdc32e8870172a30077fbeb30e647d04ee2c0d2`. The exact supplied bytes accompany both topology archives.
Build the grammar library from the unified source tree using the engine's
`zig build grammar` target and its pinned xgrammar checkout. The retained
NVFP4 archive uses the separate historical grammar library.

## Optional CUDA modules

The coherent source includes the group64 FP8-head preparation and optional BF16 latent tensor-core
prefill path; these do not require a second EXL3 executable. The runtime assets carry their distinct
hashes. All model qualification here is SM121/GB10; the presence of other architectures in a fatbin
is not an all-architecture certificate.

CUDA13.0.88 NVCC SHA256:
`b8e2347d9d7fcadf8812a27324d515de99d033d3e48611811ece7e66d7895f8a`.
The recorded host compiler is `/usr/bin/x86_64-linux-gnu-g++-13`, SHA256
`1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`.
Do not replace individual arithmetic flags with a common fast-math setting. The two exact portable
commands below intentionally differ: the FP8-head module uses `--fmad=false`, while the measured
latent-MMA command does not contain that flag.

### FP8 group64 head

```sh
nvcc -ccbin /usr/bin/x86_64-linux-gnu-g++-13 -O3 -std=c++17 --fmad=false --threads 1 --fatbin -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121 -Xptxas=-v zig/kernels/cuda/glm_head_fp8.cu -o ../glm_head_fp8_g64.fatbin
```

Expected supplied module SHA256: `0e3d6e28130225f7c7fb633d23b27f20f347e23b017599793d8e6d4e1a08a0ad`.

### BF16 latent MMA

```sh
nvcc -ccbin /usr/bin/x86_64-linux-gnu-g++-13 -O3 -std=c++17 --threads 1 --fatbin -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121 -Xptxas=-v zig/kernels/cuda/glm_latent_mma.cu -o ../glm_latent_mma.fatbin
```

Expected supplied module SHA256: `34e82c3d34080e41408fa4e2de24d229ffcbce55d3ad24bae953ffeeb9608650`.

The reference-attention cubin retains a separate Triton3.7.1 producer cache/PTX/source record.
No new Triton rebuild or bit-identical reconstruction is claimed. Source and notices for the older
EXL3 decode and borrowed gather modules are retained; a complete original producer receipt for
those historical external images was not established. The qualified recipe uses their supplied,
hash-pinned bytes. Module sources and their original include closure accompany the runtime source
packages; rebuilding a kernel requires its own contract and model gates.


## Runtime dependencies

The source distribution includes the exact patched NCCL source tree and patch,
original licenses, borrowed bridge sources and retained historical build inputs.
The NCCL tree is `e4ae48e878d15a5ba52d91a18c6b5e79b50a7db0`, based on NVIDIA commit
`12df1a11afad322be5a204a2db890161cbf8131d`. Its 1,457 source/link hashes match the
historical native build receipt. The newly packaged tar is a new archive, rather
than the historical build archive.

The borrowed bridge's old build records identify its builder image and
`/opt/zig/zig`, but do not independently pin that historical compiler executable's
hash. Related gather-kernel source and original notices are included separately;
the original external module's exact producer receipt has not been located.
These limits are retained explicitly. Qualified runs use the supplied hash-pinned
bridge and external gather module unchanged.

The portable runtime contains shared C/C++ and RDMA libraries. The image source
bundle covers 108 base/APT binary packages and the eight measured overlay package
versions, across 78 distinct source-package versions. Both source versions are
retained when the overlay replaces a library from a different base version;
base `dpkg` metadata alone does not describe the overlay. Each source package's
`.dsc` SHA-256 checks are verified. Signatures are retained but were not
independently OpenPGP-verified by the packaging helper. This source packaging is
not a claim of a bit-identical distribution-library rebuild. Original package
copyrights and the applicable license texts accompany the files.

GPU driver libraries come from the host's NVIDIA container runtime and are not
redistributed in these archives. The supplied Dockerfile documents the portable
runtime construction. The exact qualified image archive has its own checksum;
a fresh package-manager rebuild may have a different image identity.
