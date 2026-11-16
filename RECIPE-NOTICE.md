# Source and attribution

This recipe distributes a modified TensorFold engine snapshot and serving configuration for
GLM-5.3-Flash on DGX Sparks. TensorFold is the upstream engine, by the TensorFold contributors.
Its original Apache-2.0 license, notice, historical MIT notice and third-party attributions
remain in the complete source archive. The separate dependency source archives and notice
files retain their original authorship and applicable licenses.

The primary native executable reports version `1.2.12`. Its exact measured source tree,
complete source manifest, build inputs and executable hashes are recorded in the release's
`SOURCE.json` and `BUILD-RECEIPTS.json`. It contains local GLM execution, proposal-policy,
graph, expert-kernel, decision batching, native-gather integration and tiered-cache changes,
plus selected later upstream fixes. It is not an unmodified upstream release.

The supplementary retained NVFP4 configuration uses a separate `1.2.6` executable and
source snapshot, identified independently in those records. It does not inherit the primary
executable's later cache qualification. Source snapshot commits attest to measured file
bytes retrospectively; they do not claim that the historical binaries embedded those later
commit identifiers, or that a new build would be bit-identical.

MiaAI-Lab's unchanged Zig recipe is the separately configured comparison engine. Its benchmark
client sources are retained under their original attribution. No source or binary from this
recipe is inserted into that reference engine.

The primary engine's optional tensor-core latent prefill module and group-64 FP8 output-head
module are supplied separately with their complete source and compiler arguments. Their
source licenses and inherited notices remain in the corresponding TensorFold source tree.
The runtime depends on the NVIDIA CUDA driver supplied by the host; that driver is not
redistributed in the runtime overlay.

Model checkpoints are obtained separately under their providers' terms. The code license does
not replace those model licenses. The source bundle retains the original small upstream test
fixtures; it contains no deployable GLM target or drafter checkpoint.
