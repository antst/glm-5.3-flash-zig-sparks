# Building the native engine

The main release executable supports all four EXL3 projection/head options. Its version is1.2.12,
source tree is `d9d2e0f73a874c2f87445bf033efa30055f7eb5d`, and frozen source-manifest SHA256 is
`40c907eee2dc870b0744ab1e29e007b816b25c98bb65275f5919b70bc1a2ab0a`.
`SOURCE.json` maps this tree to the explicitly signed release snapshot.
A tree is not a commit ID. Source notices and original upstream attribution are retained.

The measured executable is SHA256
`9c877e091f0aa4aa8f489576b9640d346c1d1056987a2a16e9d6605823ad5259`.
Use the runtime archive for that measured identity. A rebuild has a new identity until verified and
does not inherit numerical or performance qualification just because compilation succeeds.

## Main native executable and device-test runner

Use Zig0.17.0 for Linux x86_64. Its official archive SHA256 is
`1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026`,
and the extracted compiler SHA256 is
`7c61539af19fa4082c94848f1c2b57b89e76dc6cfd5bd4c57764c3caf8f82272`.

Extract the coherent source and the146 embedded-image archive beside one another as `source/`
and `fatbins/`. From `source/`, the recorded native command with local paths substituted is:

```sh
zig build native --prefix ../build-out -j1 --summary all \
  -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=fast \
  -Dversion=1.2.12 -Dstrip=true -Dfatbins=../fatbins -Dsm=121
```

The output is `build-out/native/bin/tensorfold-native`. The test-runner target uses the same inputs:

```sh
zig build cuda-test-runner --prefix ../runner-out -j1 --summary all \
  -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=fast \
  -Dversion=1.2.12 -Dstrip=true -Dfatbins=../fatbins -Dsm=121
```

The measured runner SHA256 is
`76abf350f61c2f4a424f97ec4dabc490859da8e2557af892360af6661913a0df`.
Cross-building on x86 does not execute the AArch64 output. Device and model qualification run on
the Sparks. The source package keeps small upstream contract fixtures but no deployable target or
drafter checkpoint. The embedded-image archive contains the exact146 inputs used by this build;
optional external modules are separate runtime assets.

## Retained NVFP4 executable

The optional retained NVFP4/BF16 mode has a separate byte-verified source snapshot and executable
SHA256 `312761ce390edf72d89d72645d6ccfd9a43d151e70f50ecb3f874f17d9a8ebfd`.
Its historical version is1.2.6. Extract its source into a separate directory and use `-Dversion=1.2.6`
with the same compiler/embedded-image inputs. Do not apply the coherent source's settings or cache
qualification to that older executable. The serving renderer keeps its native binary separate and
requires NVMe off. Its release source snapshot must also be explicitly signed with the human author.

## Structured-output library

The measured runtime includes `libtfgrammar.so`, SHA-256
`abad72a824800ec45e07d2c4416bf4249266f70bf86046a0f0770653068eb09e`.
It uses xgrammar `v0.2.8`, commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`.
Clone that tagged checkout with its submodules and retain the original notices.
The recorded submodule commits are:

| Submodule | Commit |
|---|---|
| `3rdparty/cpptrace` | `6689d14c203eed390ae7bb64f56a983cfd7dff9c` |
| `3rdparty/dlpack` | `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` |
| `3rdparty/googletest` | `df1544bcee0c7ce35cd5ea0b3eb8cc81855a4140` |

The source build target is:

```sh
zig build grammar --prefix ../build-out -j1 \
  -Dtarget=aarch64-linux-gnu.2.28 -Dxgrammar=/absolute/path/to/xgrammar
```

This target requires the actual pinned Git tag and its submodules; an unpacked
directory without Git tag metadata is insufficient. The qualified library was
reused after checking all three engine shim/header files for exact equality with
its original build. The current engine's actual schema and required-tools model
checks are separate from its textual structured-answer scores.


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


## TP2 engine builds

Use the single `engine/tp2` branch for work shared by all four TP2 profiles. Its signed
commit `a6842081` has the same source tree as measured TensorFold `c829c30e`.
[Shared TP2 source](https://github.com/antst/glm-5.3-flash-zig-sparks/tree/engine/tp2)

The published runtime archives pin the signed `engine/tp2-bf16`, `engine/tp2-q4`, and
`engine/tp2-q8` source snapshots. [TP2 archive source identities](tp2/SOURCE.json) bind each to its source tree,
file manifest, native build version and measured executable. For each branch, use the same
ARM64 target, Zig toolchain and pinned 146-image input as above, with the version listed
for that branch.

The [TP2 recipe](tp2/recipe.json) declares the required external modules, libraries and
calibration files. The measured archives preserve their directory layout and executable mode.
To package a verified runtime directory from those exact files, run:

```sh
python3 -m tp2.package_runtime q4 --root /srv/glm/tp2-q4 \
  --output /tmp/glm53-tp2-q4-runtime.tar.gz
```

The TP2 and TP4 runtimes use the same ARM64 container image. Loading a published runtime
archive preserves its executable identity. Locally compiled engines have their own identity
and are measured separately.
