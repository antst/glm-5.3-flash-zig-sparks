# Third-party notices

The complete TensorFold source release preserves its original `NOTICE`,
`THIRD_PARTY_NOTICES.md` and `LICENSES/` files. Source references in those notices
are relative to that complete source tree. The primary and retained NVFP4 source
archives each preserve the notices belonging to their respective snapshots.

This recipe's `notices/` directory carries the original notices for the borrowed
native communication bridge, its external gather module, the grammar library,
the Zig runtime dependencies and the benchmark clients. See
[`notices/README.md`](notices/README.md) and
[`benchmarks/THIRD_PARTY_NOTICES.md`](benchmarks/THIRD_PARTY_NOTICES.md).

TensorFold is distributed under Apache-2.0; its pre-0.6.0 code retains its
original MIT notice. The complete source archives retain other applicable
upstream licenses, including NVIDIA, CUTLASS and PyTorch attributions where
present. This recipe's license does not replace those licenses.

The patched NCCL dependency includes its source and dual-PCI patch in
`nccl-2.32.3-dual-pci-source.tar`. The borrowed native bridge source and historical
build record are in `borrowed-native-bridge-source.tar`. The related external
module source is in `glm53-external-module-sources.tar.gz`; its exact historical
binary producer record is not established.

The ARM64 runtime image and overlay retain the installed distribution packages'
license and copyright files. Matching source-package versions are supplied in
`glm53-runtime-corresponding-sources.tar`. The host CUDA driver and model weights
are not redistributed. Obtain target and drafter checkpoints separately and
follow their providers' terms.
