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
TP2 and TP4 default to Q4 projections with an FP8 output head (`q4-fp8`).

| `--precision` | Dense projections | Output head |
|---|---|---|
| `bf16` | BF16 | BF16 |
| `q4` | Affine Q4 | BF16 |
| `q4-fp8` | Affine Q4 | FP8 E4M3 |
| `q8` | Symmetric Q8 | BF16 |

The TP2 guide describes its calibration bundles. The TP4 guide also covers the
retained NVFP4 expert option. [TP2 profiles](TP2.md#runtime) · [TP4 precision options](TP4.md#precision-options)

## Install and serve

The EXL3 serving defaults are **1,048,576 tokens per lane** and **four active requests
on TP2** or **eight on TP4**. That context includes the prompt and generated tokens.
The launch examples enable linear KV paging, optimized prefill and native NVMe snapshots. At 1M
context, all eight EXL3 profiles request an **8 GiB RAM snapshot budget per rank**,
verify that a full-window snapshot fits at startup, and allow **256 GiB of NVMe
history per rank**. Use `--learn-gib` to set the history limit to suit your disk.

1. Download the pinned model and drafter once per host and mount them read-only. [Weights and dependencies](TP4.md#weights-and-dependencies)
2. Install the shared ARM64 runtime image and the current archive for your topology and profile. The renderer checks the installed runtime against its inventory by default. [Current runtime downloads](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.3.2-tp2-tp4) · [TP2 hashes](tp2/RELEASE-ASSETS.json) · [TP4 hashes](RELEASE-ASSETS.json) · [Shared ARM64 runtime image](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.2.12-tp4)
3. Run `python3 render_config.py --tp 2` or `--tp 4` with the paths and network settings in the launch guide. Use `--verify-runtime` to check the extracted assets. The renderer writes per-rank Docker commands and cache paths into the output directory.
4. Run the generated commands on their assigned hosts. Start the followers before rank 0. Connect clients to rank 0 and use `GLM-5.3-Flash` as the API model name. Stop rank 0 first to let the followers close.

Set `GLM53_URL` to the reachable address of rank 0. Wait for `/health` to
report `"status": "ok"` and `"warming": false` before sending a request.

```sh
GLM53_URL=http://192.168.2.201:18888
curl -fsS "$GLM53_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","max_tokens":64,"messages":[{"role":"user","content":"Say hello."}]}'
```

## Performance and quality

### TP2

Our TP2 speed results use unified TensorFold `26e17d87` over direct links at traffic class 104. Our decode and `bench2` runs use eight admission slots, a 196,608-token context, the unified prefill fast path, and contiguous FP8 KV with both prompt-cache tiers off. The Mia reference uses its original Zig engine `80be4d68`. Quality scores are tied to their test revisions. [Quality table](TP2.md#quality-by-category)

#### Original Mia Zig benchmark

Rates are measured tok/s. C4, C6, and C8 use whole-wave throughput.

| Profile | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 31.19 | 49.43 | 74.09 | 89.68 | 102.81 | 884 | 8/8 · 5/8 |
| Ours Q4/BF16 | 46.01 | 64.71 | 91.23 | 107.07 | 119.09 | 881 | 8/8 · 7/8 |
| Ours Q4/FP8 | 48.94 | 74.14 | 93.32 | 111.19 | 123.88 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 40.11 | 61.09 | 86.32 | 102.13 | 115.21 | 879 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 39.40 | 49.54 | 80.32 | 95.71 | 107.28 | 872 | 8/8 · 5/8 |

#### Balanced benchmark

Fixed prompt bodies and 400 completion tokens per request.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 41.60 | 71.01 | 95.49 |
| Ours Q4/BF16 | 55.03 | 82.18 | 102.98 |
| Ours Q4/FP8 | 57.92 | 85.37 | 106.53 |
| Ours Q8/BF16 | 50.81 | 80.50 | 104.34 |
| Mia Zig Q8/BF16 | 49.17 | 75.84 | 95.25 |

[Full TP2 results](TP2.md#quality-and-performance)

### TP4

Our TP4 speed results use unified TensorFold `26e17d87` over a RoCE switch at traffic class 104. Our decode and `bench2` runs use eight admission slots, a 196,608-token context, the unified prefill fast path, and contiguous FP8 KV with both prompt-cache tiers off. The Mia reference uses its original Zig engine `80be4d68`. Quality scores are tied to their test revisions. [Quality table](TP4.md#quality-by-category)

#### Original Mia Zig benchmark

Rates are measured tok/s. C4, C6, and C8 use whole-wave throughput.

| Profile | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 53.80 | 76.98 | 127.14 | 150.19 | 173.29 | 881 | 8/8 · 5/8 |
| Ours Q4/BF16 | 76.84 | 92.32 | 149.05 | 179.48 | 198.19 | 878 | 8/8 · 7/8 |
| Ours Q4/FP8 | 80.24 | 96.16 | 158.83 | 181.59 | 202.06 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 72.37 | 93.33 | 140.87 | 168.33 | 192.94 | 878 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 63.54 | 85.16 | 129.69 | 162.56 | 175.63 | 878 | 8/8 · 5/8 |

#### Balanced benchmark

Fixed prompt bodies and 400 completion tokens per request.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 69.49 | 119.51 | 158.00 |
| Ours Q4/BF16 | 92.99 | 138.32 | 173.09 |
| Ours Q4/FP8 | 95.62 | 142.48 | 178.35 |
| Ours Q8/BF16 | 89.64 | 140.50 | 180.30 |
| Mia Zig Q8/BF16 | 80.40 | 128.70 | 166.24 |

[Full TP4 results](TP4.md#quality-and-performance)

### Matched cold prefill

Identical 62,027- and 299,992-token prompts, at the 1M context setting with one active request and no cache hits. See the topology guides for TTFT and test conditions.

| Profile | TP2 62K tok/s | TP2 300K tok/s | TP4 62K tok/s | TP4 300K tok/s |
| --- | ---: | ---: | ---: | ---: |
| BF16/BF16 | 1,651.1 | 1,614.3 | 2,593.3 | 2,587.4 |
| Q4/BF16 | 1,755.8 | 1,703.8 | 2,680.8 | 2,684.2 |
| Q4/FP8 | 1,739.5 | 1,694.1 | 2,673.6 | 2,673.3 |
| Q8/BF16 | 1,638.1 | 1,610.3 | 2,536.9 | 2,579.4 |

[Unified benchmark measurements](benchmarks/unified-measurements.json)

## Context and KV cache

Both topologies allocate active FP8 latent KV pages as requests grow.
Concurrent forks can share a resident prefix through copy-on-write. The memory
ledger queues requests if their combined reservations exceed the available
backing. Linear CUDA addresses let prefill and decode use the optimized kernels.

`--prompt-cache-gib` budgets reusable prefix snapshots in RAM. The NVMe tier
persists snapshots across restarts and restores a hit into memory before
attention resumes. Use `--prompt-cache-gib 0 --tier off` to disable the snapshot tiers
for cold-prefill measurements. Active KV still uses linear paging.

| Topology | Long-context evidence | Qualified NVMe profile |
|---|---|---|
| TP2 | All four EXL3 profiles served a cold 1,041,491-token prompt in the single-lane qualification | Q8/BF16 at 160K and 192K |
| TP4 | BF16/BF16 and Q4/BF16 served a cold 1,041,491-token prompt | BF16/BF16 at 160K and 192K |

Context, lane count, and snapshots share memory. Use `--parallel` to override
the lane count when sharing memory with other workloads. To reproduce the
throughput tables above, use the [v1.3.0 runtime and recipe](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.3.0-tp2-tp4)
with `--context 196608 --parallel 8 --prompt-cache-gib 0 --tier off`.
The topology guides give measured settings, cache restore times, and per-rank
overhead. [TP2 cache settings](TP2.md#ram-and-nvme) · [TP4 cache settings](TP4.md#ram-and-nvme-prompt-cache-tiers)

### Linear paging qualification

In v1.3.1, all eight EXL3 profiles completed the serving matrix with their default lane
limits and 1,048,576-token context. In the TP4 Q4/FP8 test, a fresh process
restored 1,046,528 tokens from NVMe and returned the first token in 10.37 s
with exact output token parity. Matched TP4 Q4/FP8 decode waves were within
0.6% of the contiguous control, with exact outputs for those requests.
[Paging qualification](benchmarks/linear-paging-qualification.json)

The v1.3.2 runtime also returns follower GPUs to idle after background NVMe
commands. A TP4 Q4/FP8 replay verified a checkpoint on each rank, idle
followers, and exact output on the cold request and its cached follow-up.
[Follower idle qualification](benchmarks/follower-idle-qualification.json)

## Source builds

Use `engine/unified` for all eight EXL3 configurations. The release archives contain the executable used for the follower idle qualification. The signed source snapshot records the exact source tree.
[Build instructions](BUILDING.md) · [Unified engine source](https://github.com/antst/glm-5.3-flash-zig-sparks/tree/engine/unified) · [TP2 archive sources](tp2/SOURCE.json) · [TP4 archive sources](SOURCE.json) · [Signatures](SIGNING.md) · [Licenses](RECIPE-NOTICE.md)
