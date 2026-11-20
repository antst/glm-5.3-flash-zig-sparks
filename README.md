# GLM-5.3-Flash on DGX Sparks

Run GLM-5.3-Flash with native Zig TensorFold, DFlash2 speculative decoding, CUDA
graph replay, and borrowed native RoCE gathers. Both topologies serve the model
as `GLM-5.3-Flash`.

## Choose a topology

| Topology | Network | Setup | Benchmarks |
|---|---|---|---|
| TP2 | Two Sparks, directly connected second ports | [TP2 launch guide](TP2.md#runtime) | [TP2 results](TP2.md#quality-and-performance) |
| TP4 | Four Sparks, first ports on a RoCE switch | [TP4 launch guide](TP4.md#serving-configuration) | [TP4 results](TP4.md#quality-and-performance) |

Both measured setups use GID 3 and RoCE traffic class 104. Each launch guide
shows the interfaces, rank order, bootstrap address, and command for its topology.

## Choose a precision

All four primary profiles use the same EXL3 4.0 bpw expert weights and FP8 live KV.

| `--precision` | Dense projections | Output head |
|---|---|---|
| `bf16` | BF16 | BF16 |
| `q4` | Affine Q4 | BF16 |
| `q4-fp8` | Affine Q4 | FP8 E4M3 |
| `q8` | Symmetric Q8 | BF16 |

The TP2 guide describes its calibration bundles. The TP4 guide also covers the
retained NVFP4 expert option. [TP2 profiles](TP2.md#runtime) · [TP4 precision options](TP4.md#precision-options)

## Install and serve

1. Download the pinned model and drafter once per host and mount them read-only. [Weights and dependencies](TP4.md#weights-and-dependencies)
2. Install the shared ARM64 runtime image and the archive for your topology and profile. Verify the archive against its inventory. [TP2 downloads](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.2.12-tp2-tp4) · [TP2 hashes](tp2/RELEASE-ASSETS.json) · [TP4 and shared downloads](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.2.12-tp4) · [TP4 hashes](RELEASE-ASSETS.json)
3. Run `python3 render_config.py --tp 2` or `--tp 4` with the paths and network settings in the launch guide. Use `--verify-runtime` to check the extracted assets. The renderer writes per-rank Docker commands and cache paths into the output directory.
4. Run the generated commands on their assigned hosts. Start the followers before rank 0. Connect clients to rank 0 and use `GLM-5.3-Flash` as the API model name. Stop rank 0 first to let the followers close.

## Performance and quality

### TP2

Our TP2 speed measurements use TensorFold `c829c30e`. The Mia row uses its original Zig engine `80be4d68`. The decode runs use the direct-link setup at traffic class 104, an eight-request limit, 196,608-token context, FP8 KV, and both prompt-cache tiers off. The quality tables identify the revisions used for scoring.

#### Original Mia Zig benchmark

Rates are measured tok/s. C4, C6, and C8 use whole-wave throughput.

| Profile | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 31.48 | 49.83 | 74.58 | 89.32 | 101.83 | 884 | 8/8 · 5/8 |
| Ours Q4/BF16 | 46.59 | 65.64 | 89.11 | 109.88 | 122.35 | 881 | 8/8 · 7/8 |
| Ours Q4/FP8 | 49.75 | 74.90 | 95.75 | 110.28 | 122.22 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 41.03 | 62.27 | 89.03 | 104.71 | 118.98 | 879 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 39.40 | 49.54 | 80.32 | 95.71 | 107.28 | 872 | 8/8 · 5/8 |

#### Balanced benchmark

Fixed prompt bodies and 400 completion tokens per request.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 41.43 | 71.47 | 96.53 |
| Ours Q4/BF16 | 55.32 | 82.84 | 103.83 |
| Ours Q4/FP8 | 58.29 | 86.42 | 107.53 |
| Ours Q8/BF16 | 51.29 | 81.45 | 105.56 |
| Mia Zig Q8/BF16 | 49.17 | 75.84 | 95.25 |

[Full TP2 results](TP2.md#quality-and-performance)

### TP4

Our TP4 speed measurements use TensorFold `c829c30e`. The Mia row uses its original Zig engine `80be4d68`. The decode runs use the switched setup at traffic class 104, an eight-request limit, 196,608-token context, FP8 KV, and both prompt-cache tiers off. The quality tables identify the revisions used for scoring.

#### Original Mia Zig benchmark

Rates are measured tok/s. C4, C6, and C8 use whole-wave throughput.

| Profile | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 54.37 | 77.91 | 128.85 | 153.51 | 169.99 | 881 | 8/8 · 5/8 |
| Ours Q4/BF16 | 76.93 | 91.99 | 154.19 | 178.31 | 197.27 | 878 | 8/8 · 7/8 |
| Ours Q4/FP8 | 80.17 | 96.13 | 152.11 | 183.88 | 200.71 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 72.60 | 93.75 | 144.95 | 168.43 | 192.79 | 878 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 63.54 | 85.16 | 129.69 | 162.56 | 175.63 | 878 | 8/8 · 5/8 |

#### Balanced benchmark

Fixed prompt bodies and 400 completion tokens per request.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 70.38 | 120.59 | 159.48 |
| Ours Q4/BF16 | 93.09 | 138.66 | 173.88 |
| Ours Q4/FP8 | 96.08 | 143.22 | 178.57 |
| Ours Q8/BF16 | 90.73 | 141.60 | 182.10 |
| Mia Zig Q8/BF16 | 80.40 | 128.70 | 166.24 |

[Full TP4 results](TP4.md#quality-and-performance)

## Context and KV cache

Both topologies keep active FP8 KV in memory. `--prompt-cache-gib` budgets reusable
prefix snapshots in RAM. The optional NVMe tier stores those snapshots across
restarts and restores a hit into memory before attention resumes. Use
`--prompt-cache-gib 0 --tier off` for cold-prefill measurements.

| Topology | Long-context evidence | Qualified NVMe profile |
|---|---|---|
| TP2 | Q8 served a cold 1,041,491-token prompt with one active lane | Q8/BF16 at 160K and 192K |
| TP4 | Completed measurements through 192K | BF16/BF16 at 160K and 192K |

Each renderer accepts up to 1,048,576 tokens. Context, lane count, and snapshots
share memory. The topology guides give measured settings, cache restore times,
and per-rank overhead. [TP2 cache settings](TP2.md#ram-and-nvme) · [TP4 cache settings](TP4.md#ram-and-nvme-prompt-cache-tiers)

## Source builds

The signed source inventories identify the code and executable shipped in each
archive. [Build instructions](BUILDING.md) · [TP2 sources](tp2/SOURCE.json) · [TP4 sources](SOURCE.json) · [Signatures](SIGNING.md) · [Licenses](RECIPE-NOTICE.md)
