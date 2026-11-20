# Frozen TensorFold refresh on direct-link TP2

The four recipe rows use the same frozen engine `c829c30e` and the profile-specific packing and head settings. All speed runs use Sparks 1 and 2, direct links, traffic class 104, EXL3 experts, FP8 KV and an eight-request limit. The benchmark context is 196,608 tokens. RAM prompt caching and NVMe were disabled. Mia’s row uses the existing measured eight-active TP2 reference.

## TP2 measured configurations

Rates are measured tok/s. C4, C6 and C8 average two waves, each measured as the summed post-first tokens divided by the common wave duration. Mia’s original clients vary the prompt content and retain natural stopping. The quality table below gives the revision used for each full suite.

| Projections / output head | Active limit | Full /959 | Text /795 | Code /164 | Small code / structured | C1 prose | C1 code | C4 prose | C6 prose | C8 prose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 8 | 884 | 730 | 154 | 8/8 · 5/8 | 31.48 | 49.83 | 74.58 | 89.32 | 101.83 |
| Ours Q4/BF16 | 8 | 881 | 721 | 160 | 8/8 · 7/8 | 46.59 | 65.64 | 89.11 | 109.88 | 122.35 |
| Ours Q4/FP8 | 8 | 882 | 722 | 160 | 8/8 · 7/8 | 49.75 | 74.90 | 95.75 | 110.28 | 122.22 |
| Ours Q8/BF16 | 8 | 879 | 724 | 155 | 8/8 · 5/8 | 41.03 | 62.27 | 89.03 | 104.71 | 118.98 |
| Mia Zig Q8/BF16 | 8 | 872 | 716 | 156 | 8/8 · 5/8 | 39.40 | 49.54 | 80.32 | 95.71 | 107.28 |

## Secondary balanced benchmark

Fixed inputs, two repetitions and 400 completion tokens per request. The median rate for each task uses the common client-clock window. The table shows the geometric mean across tasks at each concurrency.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 41.43 | 71.47 | 96.53 |
| Ours Q4/BF16 | 55.32 | 82.84 | 103.83 |
| Ours Q4/FP8 | 58.29 | 86.42 | 107.53 |
| Ours Q8/BF16 | 51.29 | 81.45 | 105.56 |
| Mia Zig Q8/BF16 | 49.17 | 75.84 | 95.25 |

Changes below use each profile’s previous measured TP2 release as its control. Exact replies match native token hashes, text and token usage on the fixed requests.

| Profile | C1 change | C4 change | C8 change | Exact replies |
| --- | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | +3.48% | +2.63% | +2.02% | 48/48 |
| Ours Q4/BF16 | +2.48% | +1.55% | +1.09% | 48/48 |
| Ours Q4/FP8 | +2.50% | +1.95% | +1.31% | 48/48 |
| Ours Q8/BF16 | +4.37% | +2.60% | +1.73% | 48/48 |

## Whole-wave throughput for the original benchmark requests

| Profile | C4 tok/s | C6 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 74.58 | 89.32 | 101.83 |
| Ours Q4/BF16 | 89.11 | 109.88 | 122.35 |
| Ours Q4/FP8 | 95.75 | 110.28 | 122.22 |
| Ours Q8/BF16 | 89.03 | 104.71 | 118.98 |
| Mia Zig Q8/BF16 | 80.32 | 95.71 | 107.28 |

## Quality by category

The table names the engine revision used for each 959-case suite. BF16 and both Q4 entries are the published quality runs. Q8 was requalified on the frozen engine. Mia’s quality run used its four-request limit.

| Profile | Quality source | Total /959 | qa /150 | reason /40 | arith /40 | track /40 | gsm8k /200 | mmlu /285 | follow /40 | legacy_humaneval_chat /164 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | `4c286e54` | 884 | 129 | 40 | 40 | 40 | 195 | 254 | 32 | 154 |
| Ours Q4/BF16 | `920e5fd2` | 881 | 128 | 40 | 39 | 40 | 195 | 249 | 30 | 160 |
| Ours Q4/FP8 | `920e5fd2` | 882 | 128 | 40 | 40 | 40 | 195 | 250 | 29 | 160 |
| Ours Q8/BF16 | `c829c30e` | 879 | 127 | 40 | 40 | 40 | 194 | 254 | 29 | 155 |
| Mia Zig Q8/BF16 | `80be4d68` | 872 | 125 | 40 | 39 | 40 | 195 | 250 | 27 | 156 |

The separate Q4 holdout scored 223/242 with the BF16 head and 225/242 with the FP8 head on the published builds. The Q8 holdout uses its own fixed inputs. The frozen Q8 build scored 223/242, the published Q8 build scored 223/242 and Mia scored 219/242.

Q8 native token-hash matches against the published Q8 runtime were 959/959 on the full suite and 242/242 on the holdout.

## Original Mia bench2 long prompts

The rates are actual prompt tokens divided by client TTFT. The original client supplies one random prompt per length.

| Profile | 16K tokens | TTFT s | Tokens/TTFT tok/s | 64K tokens | TTFT s | Tokens/TTFT tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 15802 | 10.966 | 1441.0 | 63629 | 43.619 | 1458.8 |
| Ours Q4/BF16 | 15743 | 11.112 | 1416.8 | 64082 | 44.582 | 1437.4 |
| Ours Q4/FP8 | 16018 | 11.166 | 1434.5 | 64408 | 44.481 | 1448.0 |
| Ours Q8/BF16 | 15944 | 11.150 | 1430.0 | 63739 | 44.107 | 1445.1 |
| Mia Zig Q8/BF16 | 15845 | 9.571 | 1655.5 | 63531 | 37.811 | 1680.2 |

### Q8 solo prefill and chunk width

The 4,096-row diagnostic replayed each original 2,048-row prompt three times on the same frozen binary. The 4,096-row rate is the median of the three runs. The recipe uses 2,048 rows in the speed tables.

| Prompt | 2,048 rows tok/s | 4,096 rows tok/s | Rate change | Exact native replies |
| --- | ---: | ---: | ---: | ---: |
| 16K | 1430.0 | 1515.6 | +5.99% | 3/3 |
| 64K | 1445.1 | 1521.9 | +5.32% | 3/3 |

## Original Mia bench2 mixed and paired requests

The worst p95 and maximum columns refer to inter-content gaps among the decoders.

| Profile | 50K +1 decoder TTFT s | 50K +3 decoders TTFT s | Worst p95 gap ms | Worst maximum gap ms | 30K alone TTFT s | 30K pair TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 51.10 | 51.67 | 1499 | 1764 | 20.82 | 41.87 / 22.14 |
| Ours Q4/BF16 | 51.57 | 52.06 | 1487 | 1776 | 21.06 | 42.43 / 22.46 |
| Ours Q4/FP8 | 51.59 | 52.21 | 1481 | 1767 | 21.15 | 42.37 / 22.39 |
| Ours Q8/BF16 | 51.51 | 52.21 | 1498 | 1771 | 21.00 | 42.20 / 22.32 |
| Mia Zig Q8/BF16 | 45.33 | 45.45 | 2549 | 2792 | 18.20 | 36.76 / 23.27 |

## Native tiered KV cache

These Q8 measurements use the published runtime `847757c2`. The tier restores retained KV before normal in-memory attention. The refreshed speed tests above used the tier off.

| Case | Role | Prompt tokens | Cached tokens | Native prefill s | Native TTFT s | Full request s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long160-plain | tier off | 163810 | 0 | 112.200 | 112.206 | 112.508 |
| long160-plain | tier seed | 163810 | 0 | 111.958 | 111.966 | 112.297 |
| long160-plain | NVMe reopen | 163810 | 161792 | 1.455 | 3.646 | 3.946 |
| long192-plain | tier off | 192934 | 0 | 134.180 | 134.185 | 134.487 |
| long192-plain | tier seed | 192934 | 0 | 133.233 | 138.170 | 138.495 |
| long192-plain | NVMe reopen | 192934 | 192512 | 0.496 | 2.891 | 3.193 |

[Measurement record](tp2/refresh-measurements-20261120.json)
