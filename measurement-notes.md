# Measurement definitions for the release README

These definitions accompany the published TP4 results. Every reported rate is derived from actual saved test requests and arrival clocks. No unmeasured concurrency or engine result is extrapolated.

## Balanced serving benchmark

The balanced benchmark uses four code prompts and four prose prompts. The same eight request bodies appear as solo requests, two groups of four, and one group of eight. Each measured reply contains exactly 400 completion tokens. Two repetitions are recorded for each case. Warmup requests and model-list calls are excluded from the reported rates.

For each wave, decode throughput is the sum of post-first completion tokens divided by the interval from the earliest first content arrival to the latest last content arrival. A complete 400-token stream contributes 399 post-first tokens. C4 and C8 therefore report aggregate throughput over a shared interval.

The C1 summary is the geometric mean of the eight per-prompt rates, after reducing each prompt's two repetitions. The C4 summary is the geometric mean of the two four-prompt cohorts. C8 is the measured eight-prompt cohort. These summaries must be labeled **balanced geometric mean / common-window throughput**, rather than an unspecified “C1 speed.”

For example, the recovered BF16 candidate measured:

| C1 workload | Geometric mean, tok/s |
| --- | ---: |
| Four code prompts | 82.33 |
| Four prose prompts | 53.79 |
| All eight prompts | 66.54 |

Individual code cases include LRU at 90.17 tok/s, topological sort at 83.58 tok/s, and SQL at 81.21 tok/s. These are different summaries of the same run. They do not establish an 80-to-66 tok/s regression. On the matched eight-prompt workload, the preceding BF16 controls measured 66.75–66.91 tok/s.

The balanced recorder retains exact request bytes, raw SSE, native token hashes, decoded answer/reasoning, usage and client timing intervals. Rates are recomputed from those saved client intervals. The balanced raw-SSE wrapper does not provide an independent arrival-clock sidecar, so its rate reconstruction must not be described as independent wire-clock timing.

## Serving lanes and client concurrency

C1, C4 and C8 name the number of client requests in a wave. They do not state how many requests the engine can execute simultaneously.

The controlled BF16 comparison configured both engines for eight serving lanes and disabled the prompt RAM cache. The untouched default Mia Q8 recipe uses four serving lanes, paged KV and its default 8 GiB prompt RAM cache. The proposed optimized recipe uses eight serving lanes. All comparisons disable the NVMe tier.

Mia's default Q8 balanced run measured 80.87 / 129.12 tok/s with one/four client requests. An eight-client run with its four-lane limit measured132.44 tok/s, including the queued workload. That earlier queued result is not an eight-active-lane measurement and is superseded in the README speed row by the actual eight-lane run. Every timed request reported zero cached prompt tokens. The matched comparison now explicitly runs unchanged Mia with eight serving lanes, context196608 and prompt cache0; no value is inferred from the queued four-lane run.

The closed controlled Mia BF16 reference measured 63.72 / 105.39 / 137.83 tok/s. The recovered optimized BF16 candidate measured 66.54 / 115.92 / 154.89 tok/s. These are separate sequential measurement windows with two repetitions per case. They are descriptive measurements, not a statistical guarantee.

## Original Mia clients

The original `acc.py`, `acc8.py` and `bench2.py` files are byte-identical to their Git blobs at Mia commit `80be4d6882837d8e0840827ae2aab6124fd48da3`. Their prompts, output caps, natural stopping and metric definitions are retained.

The legacy concurrency summary sums separately timed stream rates. Queued requests can occupy different time intervals, so that sum must not be presented as throughput over the whole concurrent wave.

The default four-lane Mia Q8 run illustrates the distinction:

| Client concurrency | Legacy sum of stream rates, tok/s | Shared-window rate for the same requests, tok/s |
| --- | ---: | ---: |
| C4 | 137.03 | 131.66 |
| C6 | 228.41 | 117.64 |
| C8 | 267.30 | 129.83 |

These rows retain the original clients' prompts and natural output lengths. They are not additional balanced-400 measurements. The shared-window rates use the original recorder's per-line monotonic arrival timestamps. All 63 original-client responses in this run reported zero cached prompt tokens, although the default RAM cache remained configured.

The two original clients also summarize C1 differently: `acc.py` uses the mean of two stream rates, while `bench2.py` selects the faster of two samples. Their code prompts and output lengths differ from the balanced set. Tables must retain the client and workload labels.

## Quality and precision labels

The full quality panel contains 795 diagnostic text tasks and 164 legacy chat HumanEval tasks. It is a frozen diagnostic suite, not a claim to be the complete official benchmark for each included task family. Report wrong answers, truncations and category scores alongside the total. A speed measurement does not replace the full quality panel.

“BF16,” “Q8” and “Q4” describe the target projections and head selected for that profile. Expert tensors retain their checkpoint format. EXL3 experts remain 4.0 bpw; an NVFP4 expert checkpoint remains NVFP4. Embeddings, normalization and the latent KV projection retain their declared existing paths. The selected KV-cache format is a separate option.

Cross-precision token equality is not assumed. Plain-versus-drafted and concurrent-versus-solo equality are checked within each supported profile. The recovered BF16 candidate additionally matches all 959 native output hashes of the controlled Mia BF16 reference.

The published coherent Q8 run is closed at878/959 and matches all959 native hashes of the actual-default Mia Q8 reference. SOURCE.json records the signed byte-verified source snapshots. Runtime and source identities remain unchanged by the documentation-only throughput-display correction.

## Long contexts and the tiered cache

Long-context reports must retain actual token arrays or exact request bodies, prompt-token counts, cached-token counts, cold TTFT, request elapsed time and cache configuration. A restored prefix is not a cold-prefill kernel speedup.

The tiered-cache matrix compares a cache-off reference, a fresh disk seed, a fresh-process disk reopen and a RAM-only arm at the same checkpoint, precision, input and output settings. Distinct opening and middle receipt codes prevent a constant answer from satisfying multiple long-context cases. Edited final questions and an interrupted request are checked against independent cold answers.

The intended storage behavior restores a prefix snapshot into the engine's live state; attention and decode then use that live state. Disk I/O is measured with snapshot and block-read counters. A hot RAM hit must not be described as an NVMe restore, and a declared RAM/index/staging budget must not be presented as measured memory use.

The published coherent runtime's 38-response cache matrix and model-level corruption/fault checks are closed and recorded in the release evidence. Fresh namespaces bind this executable/checkpoint/layout; an older executable's cache namespace is not reused.

## Recorded thermal samples

The final coherent BF16, Q4/BF16, Q4/FP8 and Q8/BF16 lifetimes were sampled every five seconds on all four ranks. Peak reported GPU temperatures were77,77,79 and83°C respectively. All hardware/software thermal-slowdown flags observed in those samples were inactive, and the reported thermal counter deltas were zero.

These are whole-container lifetimes, including model loading, quality cases and inter-request idle time. They are not an exact join to individual timed benchmark jobs. Memory clocks/DRAM temperatures were unavailable, and sampled flags cannot certify absence of all transient limits. Rates were not normalized by whole-lifetime clock medians, and thermal data do not establish the cause of the small BF16 secondary-benchmark deficit.

## Headline table display policy

The headline table has one Mia Q8/BF16 row: the earlier quality result, and the latest speed results with an eight-request limit. That actual run measures whole-wave C4/C6/C8=129.69/162.56/175.63 tok/s, two waves each, with all streams producing content before the first finished. Engine/source/executable/weights/precision are unchanged; context196608/cache0 match our runs, with reserve16GiB/CUDA limit100GiB explicit. C1 prose/code=63.54/85.16. The secondary fixed400 summary is80.40/128.70/166.24, reconstructed from unrounded client intervals. Raw original-client evidence is in benchmark-evidence-mia-p8. The small48 check passed8/8 code and5/8 structured with all24 plain/drafted native outputs exact; no full959 rerun or inferred P8 full-quality score is claimed. Its real eight-client/four-active queued run is retained above as a separate measurement, not substituted into that blank cell. The previously displayed267.30 was a sum of per-request rates across that queued run, not whole-wave throughput. It must not be used to rank eight-active configurations.

Use the actual Mia P8 row to compare eight-lane throughput. Our Q8 versus Mia P8 measured163.80 versus162.56 tok/s at C6 and186.18 versus175.63 at C8. Those are finite sequential measurements with original random suffixes, not a statistical guarantee or isolated transport attribution. The unchanged engine’s existing878/959 quality result is retained in the same row; a full quality rerun was unnecessary. A differing precision or active-lane limit remains explicit.
