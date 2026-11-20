# TP2 on direct-link DGX Sparks

Use two Sparks with the second ports connected directly. The measured setup uses GID 3 and traffic class 104 on the two HCA functions. Both hosts need the same model, drafter, runtime files and second-port interface names. The rank-0 direct-link IPv4 address is the NCCL and gather bootstrap address.

## Runtime

| Precision | Engine source | Runtime archive | Startup packing |
|---|---|---|---|
| BF16/BF16 | `engine/tp2-bf16` | `glm53-tp2-bf16-runtime.tar.gz` | BF16 projections and head |
| Q4/BF16 | `engine/tp2-q4` | `glm53-tp2-q4-runtime.tar.gz` | C24 activation-aware Q4, BF16 head |
| Q4/FP8 | `engine/tp2-q4` | `glm53-tp2-q4-runtime.tar.gz` | C24 activation-aware Q4, grouped FP8 head |
| Q8/BF16 | `engine/tp2-q8` | `glm53-tp2-q8-runtime.tar.gz` | Activation-aware Q8 with correlated rounding |

The Q4 and Q8 bundles include rank-specific calibration data used at weight load. Keep the
calibration and correlation directories with the matching runtime. All four profiles use the
same [EXL3 4.0 bpw weights and DFlash2 drafter](TP4.md#weights-and-dependencies).

With the selected archive extracted into `/srv/glm/tp2`, render two ranks and verify the bundle:

```sh
python3 render_config.py --tp 2 --precision q4-fp8 \
  --hosts spark1 spark2 --master 10.79.12.1 --roce-address-range 10.79.0.0/16 \
  --runtime-root /srv/glm/tp2 --model-root /srv/weights/glm53-exl3 \
  --drafter-root /srv/weights/glm53-dflash2 --cache-root /var/cache/glm53 \
  --context 196608 --parallel 8 --prompt-cache-gib 0 --tier off \
  --verify-runtime --output /tmp/glm53-tp2
```

Use `--master` and `--roce-address-range` for your direct links. The renderer uses
`enp1s0f1np1` and `rocep1s0f1` / `roceP2p1s0f1`. The HTTP, NCCL, rank-control
(NCCL port +1) and gather-bootstrap ports must be distinct. Each rank has its own
writable cache directory and the borrowed gather CPU is included in the process cpuset.

The eight-lane benchmark configuration uses a 196608-token context, stepped prefill
and 32 verification rows. The attention KV and recurrent state share the host's UMA
capacity.

The launcher accepts up to 1,048,576 tokens. Each additional lane and snapshot budget
uses memory alongside the model. The engine checks available memory at startup.

For a single long request, use `--precision q8 --context 1048576 --parallel 1
--prompt-cache-gib 0 --tier off`. The Q8/BF16 build processed a cold prompt of
1,041,491 tokens on Sparks 3 and 4 and returned the receipt code from the start of
the prompt exactly. Prefill took 781.25 seconds, about 1,333 tok/s, with drafting
disabled for that request. This is one retrieval test. Concurrent long requests and
the NVMe tier at this length still need validation.

## RAM and NVMe

The packaged TP2 profiles use contiguous FP8 live KV in memory. `--prompt-cache-gib`
budgets reusable prompt snapshots in RAM. Set `--prompt-cache-gib 0 --tier off` for
cold-prefill measurements with both prompt-cache tiers disabled.

The packaged Q8/BF16 runtime has a
measured optional NVMe tier. Set `--tier nvme --prompt-cache-gib 1.75 --learn-gib 32`
to render it with a distinct cache directory per rank and configuration under a local
NVMe cache root. A disk hit restores the snapshot into memory for subsequent attention.
`--learn-gib` limits persistent snapshots per rank. `--require-cache-tokens` controls
the minimum eligible prefix length and defaults to 137,216 tokens. This profile uses a
256 MiB staging limit and a 64 MiB index limit per rank.

The Q8 validation covered 160K/192K requests, cache reopening and checksum-failure
fallback. Per-rank pinned buffers used 207.5 MiB and the index peaked at 3.12 MiB.
With the tier on, cold 160K/192K native prefill measured 1,463/1,448 tok/s versus
1,460/1,438 with it off. After reopening a seeded cache, native time to first token
was 3.65/2.89 s on those two requests, compared with 112.21/134.18 s cold. The cold
192K time to first token with the tier enabled was 138.17 s.
The current TP2 BF16 and Q4 profiles are provided with `--tier off`. See the
[TP4 cache documentation](TP4.md#ram-and-nvme-prompt-cache-tiers) for the shared cache model.

### Paged FP8 testing

A separate TP2 Q8 build at `fe09d039` tested paged FP8 live KV with disk caching off.
It processed one 1,041,491-token prompt at 1,310 prompt tok/s and matched the native
output hash of the contiguous run. In a separate 60,952-token test, a shared RAM
prefix recovered 59,392 tokens and reduced native prefill from 41.59 s to 1.18 s.
These measurements cover the paged test build. The packaged profiles and
`c829c30e` speed runs use contiguous KV.

## Quality and performance

The [frozen TensorFold refresh tables](TP2-REFRESH.md) cover all four TP2 profiles on engine `c829c30e`. The tables below describe the packaged runtime builds.

The tables use the TP4 format. Mia's original Zig clients provide the C1, C4, C6 and C8
rates and the long-prompt diagnostics. The balanced benchmark holds 400 completion tokens
per request, uses two repetitions and reports throughput over a common client-clock window.
Speed runs use eight active lanes, FP8 KV, NVMe off and RAM prompt cache off.

<!-- BEGIN TP2 MEASUREMENTS -->

### TP2 measured configurations

Mia’s engine and default Q8/BF16 compute are unchanged. Its /959 quality score uses the earlier four-lane run and its throughput uses measured eight-lane runs. C1 rates are measured per stream. C4, C6 and C8 are total post-first tokens over the common wave, in tok/s. The original clients retain natural stopping and random prompt suffixes.

| Projections / output head | Active-lane limit | Full /959 | Text /795 | Code /164 | Small code / structured | C1 prose | C1 code | C4 prose | C6 prose | C8 prose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 8 | 884 | 730 | 154 | 8/8 / 5/8 | 29.77 | 47.20 | 70.03 | 87.97 | 104.35 |
| Ours Q4/BF16 | 8 | 881 | 721 | 160 | 8/8 / 7/8 | 45.51 | 64.49 | 91.67 | 108.66 | 115.56 |
| Ours Q4/FP8 | 8 | 882 | 722 | 160 | 8/8 / 7/8 | 48.20 | 73.15 | 94.13 | 107.37 | 124.32 |
| Ours Q8/BF16 | 8 | 879 | 724 | 155 | 8/8 / 5/8 | 38.93 | 59.46 | 85.55 | 97.60 | 116.78 |
| Mia Zig Q8/BF16 | 8 | 872 | 716 | 156 | 8/8 / 5/8 | 39.40 | 49.54 | 80.32 | 95.71 | 107.28 |

### Secondary balanced benchmark

Eight solo tasks, two four-request cohorts and one eight-request cohort. All configurations received the same 48 timed prompt bodies, with 400 completion tokens per response.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
|---|---:|---:|---:|
| Ours BF16/BF16 | 40.04 | 69.63 | 94.62 |
| Ours Q4/BF16 | 53.98 | 81.58 | 102.72 |
| Ours Q4/FP8 | 56.87 | 84.77 | 106.14 |
| Ours Q8/BF16 | 49.15 | 79.39 | 103.76 |
| Mia Zig Q8/BF16 | 49.17 | 75.84 | 95.25 |

### Quality by category

Every row uses the same 959-case inventory and unchanged scoring.

| Profile | qa /150 | reason /40 | arith /40 | track /40 | gsm8k /200 | mmlu /285 | follow /40 | legacy_humaneval_chat /164 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 129 | 40 | 40 | 40 | 195 | 254 | 32 | 154 |
| Ours Q4/BF16 | 128 | 40 | 39 | 40 | 195 | 249 | 30 | 160 |
| Ours Q4/FP8 | 128 | 40 | 40 | 40 | 195 | 250 | 29 | 160 |
| Ours Q8/BF16 | 127 | 40 | 40 | 40 | 194 | 254 | 29 | 155 |
| Mia Zig Q8/BF16 | 125 | 40 | 39 | 40 | 195 | 250 | 27 | 156 |

### Original Mia bench2 long prompts

Prompt tokens divided by client time to first token includes the first-token overhead. Each row is a single sample per length, with the original client’s random prompt suffix.

| Profile | 16K tokens | TTFT s | Tokens/TTFT tok/s | 64K tokens | TTFT s | Tokens/TTFT tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 15921 | 11.015 | 1445.3 | 64335 | 44.071 | 1459.8 |
| Ours Q4/BF16 | 16018 | 11.160 | 1435.3 | 64274 | 44.339 | 1449.6 |
| Ours Q4/FP8 | 15825 | 11.077 | 1428.7 | 63808 | 44.152 | 1445.2 |
| Ours Q8/BF16 | 15752 | 11.114 | 1417.3 | 63445 | 43.923 | 1444.5 |
| Mia Zig Q8/BF16 | 15845 | 9.571 | 1655.5 | 63531 | 37.811 | 1680.2 |

### Original Mia bench2 mixed and paired requests

A roughly 50K-token prefill runs alongside one or three decoders. Gap columns show the worst per-stream p95 and maximum inter-content gaps. Paired prompts are roughly 30K tokens.

| Profile | 50K +1 decoder TTFT s | 50K +3 decoders TTFT s | Worst p95 gap ms | Worst maximum gap ms | 30K alone TTFT s | 30K pair TTFT s |
|---|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 51.18 | 51.81 | 1500 | 1774 | 20.86 | 42.09 / 22.24 |
| Ours Q4/BF16 | 52.20 | 51.93 | 1485 | 1768 | 21.23 | 42.37 / 22.43 |
| Ours Q4/FP8 | 51.42 | 51.81 | 1489 | 1761 | 21.07 | 42.24 / 22.36 |
| Ours Q8/BF16 | 51.65 | 52.11 | 1511 | 1781 | 21.04 | 42.08 / 22.26 |
| Mia Zig Q8/BF16 | 45.33 | 45.45 | 2549 | 2792 | 18.20 | 36.76 / 23.27 |

See the [measurement record](tp2/measurements.json) for exact rates, run IDs, and per-stream mixed-request details. The runs span several sessions. Whole-profile speed differences include any differences in generated output, allocation policy and kernel scheduling.
The separate Q4 holdout scored 223/242 for C24 with the BF16 head and 225/242 with the FP8 head. C19 scored 226/242 on the same Q4 holdout. The matched Q8 holdout uses its own fixture and scored 223/242 for our Q8 and 219/242 for Mia.

<!-- END TP2 MEASUREMENTS -->
