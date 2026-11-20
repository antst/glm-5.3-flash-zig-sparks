# GLM-5.3-Flash on DGX Sparks

A native Zig TensorFold recipe for two directly connected DGX Sparks or four Sparks on a RoCE switch. It uses DFlash2 speculative decoding, graph replay and borrowed native RoCE gathers. Select the topology and precision, then render the rank configurations with the same command.

| Topology | Link | Setup and measurements |
|---|---|---|
| **TP2** | Two Sparks, direct links on the second ports, traffic class 104 | [TP2 setup and runtime bundles](TP2.md) · [Current TP2 measurements](TP2-REFRESH.md) |
| **TP4** | Four Sparks, switched first ports, traffic class 104 | [TP4 recipe and benchmark tables](TP4.md) · [Q8 refresh](TP4.md#frozen-tensorfold-refresh-q8bf16) |

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

### Render a direct-link TP2 configuration

Connect the second ports directly. The recipe uses `enp1s0f1np1`, HCAs `rocep1s0f1`
and `roceP2p1s0f1`, GID 3, and RoCE traffic class 104. Set `--master` to rank 0’s
direct-link IPv4 address. Both hosts need the matching runtime and read-only weights
at the paths below. The two Q4 profiles share one runtime archive, with a separate
head setting and cache namespace for each. [TP2 runtime bundles](tp2/RELEASE-ASSETS.json)

This example selects Q4/FP8 and uses the benchmark’s eight active lanes and cache settings. Use your real host labels, addresses, and a fresh output directory.

```sh
python3 render_config.py --tp 2 --precision q4-fp8 \
  --hosts spark1 spark2 --master 10.79.12.1 --roce-address-range 10.79.0.0/16 \
  --runtime-root /srv/glm/tp2 --model-root /srv/weights/glm53-exl3 \
  --drafter-root /srv/weights/glm53-dflash2 --cache-root /var/cache/glm53 \
  --context 196608 --parallel 8 --prompt-cache-gib 0 --tier off \
  --verify-runtime --output /tmp/glm53-tp2
```

The TP2 renderer uses stepped prefill and 32 verification rows. It writes a
`CONFIGURATION.json` with the per-rank commands, cache namespace, and source and
executable identities. Reserve distinct HTTP, NCCL bootstrap, rank-control, and
gather-bootstrap ports. The rank-control port is the NCCL port plus one. [TP2 launch details](TP2.md#runtime)

## TP2 speed and quality

These are measured direct-link rates with an eight-request limit, FP8 live KV,
196,608-token context, and both prompt-cache tiers off. The four recipe rows use
frozen engine `c829c30e`. The linked runtime archives have their own source identities
and measurement tables. Mia’s row uses the measured eight-active Zig Q8/BF16
reference. [Packaged runtime measurements](TP2.md#quality-and-performance)

On Mia’s original Zig clients, C1 is the measured per-stream rate. C4, C6, and C8 are whole-wave rates reconstructed from the client clocks. All rates are tok/s. The clients use natural stopping and vary the prompt content.

| Projections / output head | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 31.48 | 49.83 | 74.58 | 89.32 | 101.83 | 884 | 8/8 · 5/8 |
| Ours Q4/BF16 | 46.59 | 65.64 | 89.11 | 109.88 | 122.35 | 881 | 8/8 · 7/8 |
| Ours Q4/FP8 | 49.75 | 74.90 | 95.75 | 110.28 | 122.22 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 41.03 | 62.27 | 89.03 | 104.71 | 118.98 | 879 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 39.40 | 49.54 | 80.32 | 95.71 | 107.28 | 872 | 8/8 · 5/8 |

The balanced benchmark uses the same fixed prompts and 400 completion tokens per request. Q4/FP8 leads all three balanced cells. All 192 timed replies across our four profiles match their previous TP2 builds exactly.

| Profile | Balanced C1 tok/s | C4 tok/s | C8 tok/s |
|---|---:|---:|---:|
| Ours BF16/BF16 | 41.43 | 71.47 | 96.53 |
| Ours Q4/BF16 | 55.32 | 82.84 | 103.83 |
| Ours Q4/FP8 | 58.29 | 86.42 | 107.53 |
| Ours Q8/BF16 | 51.29 | 81.45 | 105.56 |
| Mia Zig Q8/BF16 | 49.17 | 75.84 | 95.25 |

Q8 scored 879/959 and 223/242 on the frozen engine, with all 1,201 native token
hashes matching the packaged Q8 build. The BF16, both Q4, and Mia quality scores
come from prior runs. The Q4 holdout scores were 223/242 with a BF16 head and
225/242 with an FP8 head. [Quality by category and source](TP2-REFRESH.md#quality-by-category)

At roughly 64K, the original client measured 1,445 prompt tok/s for our Q8 and 1,680 for Mia’s Q8 using prompt tokens divided by time to first token. The separate 4,096-row solo replay raised our Q8 rate to 1,522 tok/s with exact replies. [Long-prompt, mixed, and paired-request measurements](TP2-REFRESH.md#original-mia-bench2-long-prompts)

## TP2 long context and KV cache

- **Context and lanes.** The renderer accepts up to **1,048,576 tokens** and up to eight lanes. Both consume shared memory, and the runtime checks available memory at startup. The tested single-long-request setup uses `--precision q8 --context 1048576 --parallel 1 --prompt-cache-gib 0 --tier off`. It processed a cold 1,041,491-token prompt at about 1,333 prompt tok/s and returned the expected retrieval code. [Long-context setup](TP2.md#runtime)
- **Live KV and RAM prefix reuse.** Inference uses FP8 KV in memory. The packaged TP2 profiles use contiguous live KV. `--prompt-cache-gib` budgets reusable prompt snapshots in RAM. Use `--prompt-cache-gib 0 --tier off` to measure cold prefill.
- **Persistent NVMe snapshots.** On TP2, the renderer enables the native disk tier for Q8/BF16. Use `--tier nvme --prompt-cache-gib 1.75 --learn-gib 32` with the cache root on local NVMe. A hit restores the saved prefix into memory before attention resumes. `--learn-gib` sets the disk budget per rank, and `--require-cache-tokens` sets the minimum eligible prefix length (default 137,216 tokens). The renderer gives each rank and configuration its own cache directory.
- **Measured disk behavior.** On the packaged Q8 build, NVMe reopen reduced native time to first token from **112.21 s to 3.65 s** at 160K and **134.18 s to 2.89 s** at 192K. The run used 207.5 MiB of pinned staging per rank and peaked at 3.12 MiB of index metadata per rank. Validation covered cold seeding at both lengths and recovery from a corrupted snapshot. Disk-tier operation at 1M is still unverified. [TP2 cache configuration](TP2.md#ram-and-nvme) · [TP2 cache timings](TP2-REFRESH.md#native-tiered-kv-cache) · [TP4 cache timings](TP4.md#native-tiered-kv-cache-release-measurements)

## TP4 Q8 refresh

Frozen TensorFold `c829c30e` ran Q8/BF16 on four switched Sparks at traffic class
104, with eight active lanes and both prompt-cache tiers off. The reference is
the saved packaged TP4 Q8/BF16 run at `1e3ba2cf`.

| Balanced decode | Packaged tok/s | Refresh tok/s | Change |
|---|---:|---:|---:|
| C1 | 84.92 | 90.31 | +6.34% |
| C4 | 135.92 | 141.66 | +4.23% |
| C8 | 176.67 | 181.63 | +2.81% |

All 48 fixed-prompt replies matched exactly. The same 63,997-token prompt
produced an exact reply and took 31.60 s to prefill on the packaged build and
31.63 s on the refresh. [TP4 refresh benchmark details](TP4.md#frozen-tensorfold-refresh-q8bf16)
