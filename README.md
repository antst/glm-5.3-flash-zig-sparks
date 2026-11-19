# GLM-5.3-Flash on DGX Sparks

A native Zig TensorFold recipe for two directly connected DGX Sparks or four Sparks on a RoCE switch. It uses DFlash2 speculative decoding, graph replay and borrowed native RoCE gathers. Select the topology and precision, then render the rank configurations with the same command.

| Topology | Link | Setup and measurements |
|---|---|---|
| **TP2** | Two Sparks, direct links on the second ports, traffic class 104 | [TP2 recipe and benchmark tables](TP2.md) |
| **TP4** | Four Sparks, switched first ports, traffic class 104 | [TP4 recipe and benchmark tables](TP4.md) |

## Choose a precision

Both topologies use the same EXL3 4.0 bpw expert checkpoint and FP8 attention KV storage. The projection format and output head are independent.

| `--precision` | Dense projections | Output head |
|---|---|---|
| `bf16` | BF16 | BF16 |
| `q4` | Affine Q4 | BF16 |
| `q4-fp8` | Affine Q4 | FP8 E4M3 |
| `q8` | Symmetric Q8 | BF16 |

TP2 uses the C24 calibration for both Q4 profiles and correlation-aware packing for Q8. The TP4 page also documents the retained NVFP4 expert option.

## Run the recipe

1. Use the [pinned weights and drafter revisions](TP4.md#weights-and-dependencies). Keep one read-only copy of each checkpoint per host.
2. Install the hash-pinned [TP2](tp2/RELEASE-ASSETS.json) or [TP4](RELEASE-ASSETS.json) runtime bundle for your selected profile.
3. Run `python3 render_config.py --tp 2` or `--tp 4` with the paths and network settings on the topology page. The renderer writes rank payloads and commands locally.
4. Create the cache directories and start the ranks using the rendered commands. Start followers before rank 0 and stop rank 0 first.

For source builds, see [Building](BUILDING.md). Each runtime pins its signed [TP2](tp2/SOURCE.json) or [TP4](SOURCE.json) engine snapshot and the exact executable used for measurement. Code, model and drafter licenses are covered in [Notices](RECIPE-NOTICE.md).
