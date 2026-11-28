# TP2 on direct-link DGX Sparks

Use two Sparks with the second ports connected directly. The measured setup uses GID 3 and traffic class 104 on the two HCA functions. Both hosts need the same model, drafter, runtime files and second-port interface names. The rank-0 direct-link IPv4 address is the NCCL and gather bootstrap address.

## Runtime

Build and develop all four profiles from [`engine/unified`](https://github.com/antst/glm-5.3-flash-zig-sparks/tree/engine/unified).
Exact sources for the published runtime archives are identified by signed tags.
[Archive sources](tp2/SOURCE.json)

| Precision | Archive source commit | Runtime archive | Startup packing |
|---|---|---|---|
| BF16/BF16 | `75429cd5` | `glm53-tp2-bf16-paged-runtime.tar.gz` | BF16 projections and head |
| Q4/BF16 | `75429cd5` | `glm53-tp2-q4-paged-runtime.tar.gz` | C24 activation-aware Q4, BF16 head |
| Q4/FP8 (default) | `75429cd5` | `glm53-tp2-q4-paged-runtime.tar.gz` | C24 activation-aware Q4, grouped FP8 head |
| Q8/BF16 | `75429cd5` | `glm53-tp2-q8-paged-runtime.tar.gz` | Activation-aware Q8 with correlated rounding |

The Q4 and Q8 bundles include rank-specific calibration data used at weight load. Keep the
calibration and correlation directories with the matching runtime. All four profiles use the
same [EXL3 4.0 bpw weights and DFlash2 drafter](TP4.md#weights-and-dependencies).

Download the matching archive and verify its SHA256 before extracting it.
[TP2 runtime downloads](https://github.com/antst/glm-5.3-flash-zig-sparks/releases/tag/v1.3.2-tp2-tp4) · [Archive hashes](tp2/RELEASE-ASSETS.json)

Extract the Q4 archive into `/srv/glm/tp2`, then render two ranks and verify the
bundle. The serving defaults are Q4/FP8, four active lanes with 1,048,576 tokens
each, linear KV paging, optimized prefill, and native NVMe snapshots. For another profile, use its runtime archive and pass `--precision bf16`,
`--precision q4`, or `--precision q8`.

```sh
python3 render_config.py --tp 2 \
  --hosts spark1 spark2 --master 10.79.12.1 --roce-address-range 10.79.0.0/16 \
  --runtime-root /srv/glm/tp2 --model-root /srv/weights/glm53-exl3 \
  --drafter-root /srv/weights/glm53-dflash2 --cache-root /var/cache/glm53 \
  --verify-runtime --output /tmp/glm53-tp2
```

Use `--master` and `--roce-address-range` for your direct links. The renderer uses
`enp1s0f1np1` and `rocep1s0f1` / `roceP2p1s0f1`. The HTTP, NCCL, rank-control
(NCCL port +1) and gather-bootstrap ports must be distinct. Each rank has its own
writable cache directory and the borrowed gather CPU is included in the process cpuset.

The historical throughput benchmark uses the v1.3.0 recipe with
`--context 196608 --parallel 8 --prompt-cache-gib 0 --tier off`. Both the current
serving recipe and the benchmark use stepped prefill and 32 verification rows.
The attention KV and recurrent state share the host's UMA capacity.

Each additional lane and snapshot budget uses memory alongside the model. The
engine checks available memory at startup.

In the single-lane qualification, each of the four EXL3 profiles processed a cold
1,041,491-token prompt and retrieved the receipt code from the start of the prompt
exactly. All four returned identical answer token IDs. Prefill took 778 to 786
seconds on Sparks 3 and 4. The retrieval request disabled drafting. The NVMe
measurements below cover 160K and 192K prompts.

## Quality and performance

Our TP2 speed results use unified TensorFold `26e17d87` over direct links at traffic class 104. Our decode and `bench2` runs use eight admission slots, a 196,608-token context, the unified prefill fast path, and contiguous FP8 KV with both prompt-cache tiers off. The Mia reference uses its original Zig engine `80be4d68`. Quality scores are tied to their test revisions. [Quality table](TP2.md#quality-by-category)

### Original Mia Zig benchmark

C1 shows per-stream decode rates. C4, C6, and C8 show whole-wave throughput. All rates are in tok/s.

| Profile | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 31.19 | 49.43 | 74.09 | 89.68 | 102.81 | 884 | 8/8 · 5/8 |
| Ours Q4/BF16 | 46.01 | 64.71 | 91.23 | 107.07 | 119.09 | 881 | 8/8 · 7/8 |
| Ours Q4/FP8 | 48.94 | 74.14 | 93.32 | 111.19 | 123.88 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 40.11 | 61.09 | 86.32 | 102.13 | 115.21 | 879 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 39.40 | 49.54 | 80.32 | 95.71 | 107.28 | 872 | 8/8 · 5/8 |

### Balanced benchmark

The same fixed prompts run twice at each concurrency, with 400 completion tokens per request. Rates use the common client-clock window.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 41.60 | 71.01 | 95.49 |
| Ours Q4/BF16 | 55.03 | 82.18 | 102.98 |
| Ours Q4/FP8 | 57.92 | 85.37 | 106.53 |
| Ours Q8/BF16 | 50.81 | 80.50 | 104.34 |
| Mia Zig Q8/BF16 | 49.17 | 75.84 | 95.25 |

### Whole-wave throughput

| Profile | C4 tok/s | C6 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 74.09 | 89.68 | 102.81 |
| Ours Q4/BF16 | 91.23 | 107.07 | 119.09 |
| Ours Q4/FP8 | 93.32 | 111.19 | 123.88 |
| Ours Q8/BF16 | 86.32 | 102.13 | 115.21 |
| Mia Zig Q8/BF16 | 80.32 | 95.71 | 107.28 |

### Quality by category

Each row uses the same 959-case inventory.

| Profile | Engine | Full /959 | qa /150 | reason /40 | arith /40 | track /40 | gsm8k /200 | mmlu /285 | follow /40 | legacy_humaneval_chat /164 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | `4c286e54` | 884 | 129 | 40 | 40 | 40 | 195 | 254 | 32 | 154 |
| Ours Q4/BF16 | `920e5fd2` | 881 | 128 | 40 | 39 | 40 | 195 | 249 | 30 | 160 |
| Ours Q4/FP8 | `920e5fd2` | 882 | 128 | 40 | 40 | 40 | 195 | 250 | 29 | 160 |
| Ours Q8/BF16 | `c829c30e` | 879 | 127 | 40 | 40 | 40 | 194 | 254 | 29 | 155 |
| Mia Zig Q8/BF16 | `80be4d68` | 872 | 125 | 40 | 39 | 40 | 195 | 250 | 27 | 156 |

Q4 scored 223/242 with the BF16 head and 225/242 with the FP8 head on its holdout. On the Q8 holdout, TensorFold scored 223/242 and Mia scored 219/242.

### Original Mia bench2 long prompts

Each cell uses the measured prompt length and client time to first token. The client generates a different prompt per length.

| Profile | 16K tokens | TTFT s | Prompt tok/s | 64K tokens | TTFT s | Prompt tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 15710 | 9.746 | 1612.0 | 63135 | 38.253 | 1650.4 |
| Ours Q4/BF16 | 16075 | 9.539 | 1685.2 | 64338 | 37.147 | 1732.0 |
| Ours Q4/FP8 | 16067 | 9.555 | 1681.6 | 64484 | 37.437 | 1722.5 |
| Ours Q8/BF16 | 15742 | 9.864 | 1595.9 | 63840 | 38.963 | 1638.5 |
| Mia Zig Q8/BF16 | 15845 | 9.571 | 1655.5 | 63531 | 37.811 | 1680.2 |

### Original Mia bench2 mixed and paired requests

A roughly 50K-token prefill runs beside one or three decoders. Gaps are the worst per-stream values. The paired prompts are roughly 30K tokens each.

| Profile | 50K +1 decoder TTFT s | 50K +3 decoders TTFT s | Worst p95 gap ms | Worst maximum gap ms | 30K alone TTFT s | 30K pair TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 45.92 | 46.31 | 1357 | 1624 | 18.37 | 37.10 / 21.16 |
| Ours Q4/BF16 | 44.94 | 45.33 | 1299 | 1577 | 17.84 | 17.85 / 35.93 |
| Ours Q4/FP8 | 45.85 | 45.35 | 1295 | 1597 | 18.03 | 36.23 / 22.98 |
| Ours Q8/BF16 | 46.79 | 47.16 | 1363 | 1643 | 18.63 | 18.62 / 37.54 |
| Mia Zig Q8/BF16 | 45.33 | 45.45 | 2549 | 2792 | 18.20 | 36.76 / 23.27 |

[Earlier measurement record](tp2/refresh-measurements-20261120.json)

### Matched cold prefill

The same 62,027- and 299,992-token requests run one at a time at the full 1,048,576-token context. Every request retrieved the expected receipt code with zero cache hits. Rates use the engine's prefill timer. TTFT uses the client clock. Paging, RAM prompt caching, and NVMe caching are disabled. BF16/BF16 ran on Sparks 1 and 2. Q4/BF16, Q4/FP8, Q8/BF16 ran on Sparks 3 and 4.

| Profile | 62K tok/s | 62K TTFT s | 300K tok/s | 300K TTFT s |
| --- | ---: | ---: | ---: | ---: |
| BF16/BF16 | 1,651.1 | 37.59 | 1,614.3 | 185.92 |
| Q4/BF16 | 1,755.8 | 35.36 | 1,703.8 | 176.16 |
| Q4/FP8 | 1,739.5 | 35.68 | 1,694.1 | 177.16 |
| Q8/BF16 | 1,638.1 | 37.89 | 1,610.3 | 186.38 |

[Unified benchmark measurements](benchmarks/unified-measurements.json)

## RAM and NVMe

The packaged TP2 profiles allocate FP8 latent KV pages on demand. TensorFold
sets the memory reserve automatically and can allocate additional private
sequence buffers on admission. `--prompt-cache-gib`
budgets reusable prompt snapshots in RAM. Set `--prompt-cache-gib 0 --tier off` for
cold-prefill measurements with both prompt-cache tiers disabled.

Native NVMe snapshots are enabled by default for all four EXL3 profiles. Point
`--cache-root` at a local NVMe filesystem. The renderer uses a distinct directory
per rank and configuration. `--learn-gib` defaults to 256 GiB per rank.
`--require-cache-tokens` requires a full-window snapshot to fit in the effective
RAM pool at startup by default. The 1M default requests 8 GiB per rank. The NVMe
worker uses a 256 MiB staging limit and a 64 MiB index limit per rank. A disk hit
restores a snapshot into memory for subsequent attention. Use `--tier off` to
disable persistent snapshots.

The historical Q8 validation covered 160K/192K requests, cache reopening and checksum-failure
fallback. Per-rank pinned buffers used 207.5 MiB and the index peaked at 3.12 MiB.
With the tier on, cold 160K/192K native prefill measured 1,463/1,448 tok/s versus
1,460/1,438 with it off. After reopening a seeded cache, native time to first token
was 3.65/2.89 s on those two requests, compared with 112.21/134.18 s cold. The cold
192K time to first token with the tier enabled was 138.17 s.
See the
[TP4 cache documentation](TP4.md#ram-and-nvme-prompt-cache-tiers) for the shared cache model.

### Historical paged FP8 testing

A separate TP2 Q8 build at `fe09d039` tested paged FP8 live KV with disk caching off.
It processed one 1,041,491-token prompt at 1,310 prompt tok/s and matched the native
output hash of the contiguous run. In a separate 60,952-token test, a shared RAM
prefix recovered 59,392 tokens and reduced native prefill from 41.59 s to 1.18 s.
These measurements cover the older paged test build. The `c829c30e` speed
runs used contiguous KV. [Current paging qualification](README.md#linear-paging-qualification)

### Native tiered KV cache

The historical Q8 runtime `847757c2` supplied the cold, seed, and reopen timings.

| Case | Role | Prompt tokens | Cached tokens | Native prefill s | Native TTFT s | Full request s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long160-plain | tier off | 163810 | 0 | 112.200 | 112.206 | 112.508 |
| long160-plain | tier seed | 163810 | 0 | 111.958 | 111.966 | 112.297 |
| long160-plain | NVMe reopen | 163810 | 161792 | 1.455 | 3.646 | 3.946 |
| long192-plain | tier off | 192934 | 0 | 134.180 | 134.185 | 134.487 |
| long192-plain | tier seed | 192934 | 0 | 133.233 | 138.170 | 138.495 |
| long192-plain | NVMe reopen | 192934 | 192512 | 0.496 | 2.891 | 3.193 |
