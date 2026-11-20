# GLM-5.3-Flash on switched DGX Sparks

TP4 release source: `1e3ba2cf32cb1dea035e38ad41adffc7ea402c6d`. All four main projection/head modes use the same qualified native executable.

The inference engine is native Zig TensorFold with DFlash2, graph replay for the qualified FFN and
drafter paths, batched decisions, retained sequence state, GPU index kernels and borrowed native
RoCE gathers. TP4 uses four GB10 Sparks through the switch with traffic class104.

## Precision options

All four main options use the same EXL34.0bpw expert checkpoint. Projection weights, the target
output head and attention-cache storage are independent precision choices.

| Option | Target projections | Target output head | Attention KV storage |
|---|---|---|---|
| `bf16` | BF16 | BF16 | FP8 |
| `q4` | affine Q4, group64 | BF16 | FP8 |
| `q4-fp8` | affine Q4, group64 | FP8 E4M3, group64/FP32 scales | FP8 |
| `q8` | symmetric Q8, group64/F16 scales | BF16 | FP8 |

Experts keep their checkpoint encoding, and the drafter keeps its pinned preparation path. The
latent KV projection retains BF16. The KV storage format does not have to match the output head.
FP8 KV storage is quantized and can change answers compared with BF16 KV; the tables hold KV at FP8 throughout.
Changing projection or head precision can change answers; compare the quality categories as well
as the total. Within-mode served/raw and plain/drafted checks are separate from cross-mode quality.

The retained `nvfp4-bf16` option uses NVFP4 experts and BF16 projections/head with FP8 KV. It has a
separate measured executable and supplementary results. Its older executable is not qualified for
the current NVMe worker; the renderer requires `--tier off` for this retained option. Failed NVFP4
Q4 experiments are not serving options. Further NVFP4 tuning is parked.

## Weights and dependencies

Download each checkpoint once per host. Mount that read-only directory into all matching serving
configurations; changing projection/head options does not require another expert checkpoint copy.
Keep writable cache roots separate from the weight repository. Checkpoint revisions are immutable;
use a new revision directory rather than editing pinned weight files in place.

| Component | Repository | Pinned revision |
|---|---|---|
| EXL3 experts | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` | `078455ffe6472f9a52fbc1139f58b9db2881b25c` |
| Optional NVFP4 experts | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | `caca4e6a4ebbd66f159d3d2fc256683fd6e27177` |
| DFlash2 drafter | `incoai/GLM-5.3-Flash-DFlash2` | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |

Source and model licenses are separate. Read the pinned model cards before use. The
[DFlash2 card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/README.md#license)
specifies CC BY-NC-ND4.0 for research/evaluation and separate commercial licensing from its provider.
Weights are downloaded separately and are not redistributed in the runtime package.

## Quality and performance

The fixed quality suite contains795 text tasks and164 coding tasks. The small panel contains eight
coding and eight structured-output tasks, with separate served/raw and plain/drafted checks. Wrong
answers and truncations stay in the results. Q4 selection requires8/8 coding and at least7/8
structured results. Q8 is checked against the actual-default Mia Q8 reference, including its native
output hashes, rather than changing that reference's results to fit a different precision floor.

The release evidence bundle contains the original benchmark request bodies, byte-exact SSE
responses and recorded line-arrival clocks for all five rows. Run
`python3 benchmark-evidence/verify.py benchmark-evidence` to check the request/SSE/client hashes
and independently recompute both C4/C6/C8 metrics. This needs neither a model nor a GPU. Original
clients and their licenses are included; the benchmark requests and natural stopping are unchanged.
The separately measured Mia eight-lane run is in `benchmark-evidence-mia-p8`; run
`python3 benchmark-evidence-mia-p8/verify.py benchmark-evidence-mia-p8` to verify its41 original-client responses.

### Frozen TensorFold refresh Q8/BF16

The November 20 speed check used frozen TensorFold `c829c30e` and the published
TP4 Q8/BF16 recipe on four switched Sparks. It retained the EXL3 4.0 bpw experts,
Q8 projections, BF16 head, DFlash2 drafter, contiguous FP8 KV, and borrowed dual
RoCE rails at traffic class 104. The context was 196,608 tokens, the active limit
was eight requests, and both the RAM prompt cache and NVMe were off. The
comparison uses the saved packaged run at `1e3ba2cf`.

| Balanced decode | Packaged tok/s | Refresh tok/s | Change |
|---|---:|---:|---:|
| C1 | 84.92 | 90.31 | +6.34% |
| C4 | 135.92 | 141.66 | +4.23% |
| C8 | 176.67 | 181.63 | +2.81% |

The two runs use the same fixed prompt bodies, two repetitions, and 400
completion tokens per response. All 48 native token hashes, text hashes, and
token-usage signatures matched. The saved 63,997-token cold prompt also
matched its native output. Native prefill measured 31.604 s on the packaged
build and 31.630 s on the refresh, about 2,025 and 2,023 prompt tok/s.

The packaged-runtime tables below give the full quality scores, original
client results, and other precision profiles. [Refresh measurement record](measurements.json)

<!-- BEGIN CURRENT MEASUREMENTS -->

### TP4 measured configurations

Every populated speed cell below comes from actual runs of Mia’s unchanged `acc.py`, `acc8.py` and `bench2.py`, with the saved request bodies and SSE arrival clocks. **There are no extrapolated rates.** C1 is the original measured stream rate; C4/C8 use aggregate post-first tokens over the entire wave, not a sum of separately timed stream rates.

**All five rows below use an eight-request limit for these speed measurements.** Mia’s engine, weights and Q8/BF16 compute are unchanged; its latest speed run uses context196608 and prompt cache0, matching ours. Its existing quality score is retained. Q8/BF16 is the same-precision reference pair.

All rows use EXL3 experts and FP8 KV. NVMe and the prompt RAM cache are off for the speed measurements. This is a whole-configuration comparison, not an isolated-kernel comparison.

| Projections / output head | Active-lane limit | Full /959 | Text /795 | Code /164 | Small code / structured | 1 active: prose | 1 active: code | 4 active: prose | 8 active: prose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 8 | 881 | 725 | 156 | 8/8 / 5/8 | 51.25 | 73.53 | 124.15 | 170.43 |
| Ours Q4/BF16 | 8 | 878 | 724 | 154 | 8/8 / 7/8 | 72.60 | 87.12 | 149.04 | 190.15 |
| Ours Q4/FP8 | 8 | 882 | 729 | 153 | 8/8 / 7/8 | 75.85 | 91.50 | 148.15 | 200.53 |
| Ours Q8/BF16 | 8 | 878 | 724 | 154 | 8/8 / 5/8 | 68.30 | 88.37 | 145.11 | 186.18 |
| Mia Zig Q8/BF16 | 8 | 878 | 724 | 154 | 8/8 / 5/8 | 63.54 | 85.16 | 129.69 | 175.63 |

Rates are tok/s. Mia’s row combines the existing878/959 quality result with its latest measured eight-request speed run; the unchanged engine/weights/precision did not require a full quality rerun. The short panel again passed8/8 code and5/8 structured, with all24 plain/drafted native outputs exact. Its speed run used reserve16GiB and a100GiB CUDA memory ceiling. The original client’s request-rate sum is retained in the raw evidence and described in measurement-notes.md; it is excluded from the headline comparison because queued requests can occupy different time intervals.

Coherent BF16 original-client speed and small panel are closed on the release executable. Its full881/959 quality panel is closed. Q8 full878/959 and all959 native hashes match actual-default Mia. Recovered BF16 similarly matches all959 native hashes of source-unmodified controlled Mia BF16. EXL3 Q4 and Q4/FP8 meet the8/8 code and7/8 structured floor. Mia Q8 and ours both score8/8 code and5/8 structured; all48 small-panel native outputs match.

## Secondary balanced benchmark

400 completion tokens, two repetitions; eight solo-task geometric mean, two four-request cohorts and one eight-request cohort using common-window throughput. The Mia row uses the latest actual eight-lane measurement with context196608/cache0. These values do not replace Mia’s original benchmark.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
|---|---:|---:|---:|
| Ours BF16/BF16 | 66.41 | 115.71 | 155.07 |
| Ours Q4/BF16 | 88.28 | 134.26 | 169.21 |
| Ours Q4/FP8 | 91.03 | 138.69 | 175.36 |
| Ours Q8/BF16 | 84.92 | 135.92 | 176.67 |
| Mia Zig Q8/BF16 | 80.40 | 128.70 | 166.24 |

Previous matched BF16 balanced controls:66.75–66.91 /116.26–116.34 /155.39–155.42. That small secondary speed deficit remains open. Signed byte-verified source snapshot commits and their limits are recorded in SOURCE.json. Retained NVFP4 and controlled BF16 records remain in supplementary data; further NVFP4 optimization is stopped per the user’s direction.

Precision labels are independent: EXL34.0bpw describes expert weights, the first option names dense projection weights, the second names the target output head, and FP8 KV describes attention-cache storage. KV and head precision do not have to match.

## Same original benchmark requests: whole-wave throughput

These rates are reconstructed from the actual acc.py/acc8.py requests and recorded line-arrival clocks: summed post-first tokens divided by the interval from the first content in the wave to the last. Natural stopping and prompt bodies are unchanged. The Mia result includes two actual waves at C6 and two at C8 with an eight-request limit. Every stream produced content before the first stream finished in each wave.

| Profile | 4 active tok/s | 6 active tok/s | 8 active tok/s |
|---|---:|---:|---:|
| Ours BF16/BF16 | 124.15 | 147.69 | 170.43 |
| Ours Q4/BF16 | 149.04 | 175.30 | 190.15 |
| Ours Q4/FP8 | 148.15 | 175.80 | 200.53 |
| Ours Q8/BF16 | 145.11 | 163.80 | 186.18 |
| Mia Zig Q8/BF16 | 129.69 | 162.56 | 175.63 |

## Quality by category

Same frozen inventory and unchanged scorers. Scores are finite diagnostics, not a guarantee of identical answers or universal superiority across precisions.

| Profile | qa /150 | reason /40 | arith /40 | track /40 | gsm8k /200 | mmlu /285 | follow /40 | legacy_humaneval_chat /164 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 129 | 40 | 40 | 40 | 195 | 253 | 28 | 156 |
| Ours Q4/BF16 | 127 | 40 | 40 | 39 | 195 | 250 | 33 | 154 |
| Ours Q4/FP8 | 128 | 40 | 40 | 39 | 195 | 252 | 35 | 153 |
| Ours Q8/BF16 | 131 | 40 | 39 | 40 | 196 | 249 | 29 | 154 |
| Mia Zig Q8/BF16 | 131 | 40 | 39 | 40 | 196 | 249 | 29 | 154 |

## Original Mia bench2: long prompts

Cache disabled. Counts are actual prompt tokens. The original benchmark reports prompt tokens divided by client TTFT; this includes first-token overhead and is not native GPU prefill duration. Request-specific nonces and finite single samples are retained.

| Profile | 16K tokens | TTFT s | Tokens/TTFT tok/s | 64K tokens | TTFT s | Tokens/TTFT tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Ours BF16/BF16 | 15978 | 7.891 | 2024.9 | 63750 | 31.285 | 2037.7 |
| Ours Q4/BF16 | 16163 | 8.170 | 1978.3 | 64723 | 32.220 | 2008.8 |
| Ours Q4/FP8 | 15901 | 7.977 | 1993.3 | 63817 | 31.776 | 2008.3 |
| Ours Q8/BF16 | 15923 | 8.002 | 1990.0 | 63997 | 31.622 | 2023.8 |
| Mia Zig Q8/BF16 | 16046 | 8.980 | 1786.9 | 64596 | 33.867 | 1907.4 |

## Original Mia bench2: mixed and paired requests

A roughly50K-token prefill runs alongside one or three decoders. The gap column is the worst per-stream p95 and maximum content-arrival gaps reported by the original harness, in milliseconds. The paired case uses roughly30K-token prompts. These are observed client latencies, not additive GPU stage timings. Complete counts and per-stream data are in measurements.json.

| Profile | 50K +1 decoder TTFT s | 50K +3 decoders TTFT s | Worst p95 gap ms | Worst maximum gap ms | 30K alone TTFT s | 30K pair TTFT s |
|---|---:|---:|---:|---:|---:|---|
| Ours BF16/BF16 | 36.76 | 36.99 | 1062 | 1298 | 15.14 | 30.11 / 17.09 |
| Ours Q4/BF16 | 37.97 | 38.28 | 1067 | 1302 | 15.51 | 16.39 / 31.08 |
| Ours Q4/FP8 | 37.09 | 37.39 | 1062 | 1285 | 15.11 | 16.10 / 30.46 |
| Ours Q8/BF16 | 37.14 | 37.45 | 1068 | 1306 | 15.21 | 16.06 / 30.59 |
| Mia Zig Q8/BF16 | 44.07 | 44.37 | 77 | 2610 | 16.39 | 33.56 / 16.38 |

The compared sources use2048 prompt rows per busy pass in our build and4096 in Mia. Mia's lower p95 coexists with a larger maximum pause, while our long-prefill TTFT is shorter. These whole-configuration observations do not isolate the causal contribution of chunk size; neither percentile alone establishes universally better responsiveness.


<!-- END CURRENT MEASUREMENTS -->

The five rows compare our four EXL3 options with source-unmodified Mia Zig.
Mia uses EXL3/Q8/BF16 and paged context; ours uses nonpaged context. Both use an eight-request limit
for the current acc.py/acc8.py and balanced speed results. The older bench2 long/mixed results retain
Mia’s original four-request limit; those workloads use at most four concurrent requests.
The original benchmark clients are unchanged on both runtimes.
A sum of separately timed stream rates can include queued waves; it is not automatically whole-wave
capacity. The original metric remains visible, with the common-window result for the same requests
reported separately. All timed request cache hits were checked; NVMe is disabled for speed runs.

`benchmarks/mia/acc.py`, `acc8.py` and `bench2.py` preserve the original client source. The balanced
400-token diagnostic is secondary. Full requests, raw SSE, per-category scores, executable hashes
and finite-repeat limits belong in the accompanying machine-readable results and ledger. Timing
results are from the stated hardware and settings, not a universal performance guarantee.

## Serving configuration

The release provides hash-pinned ARM64 runtime assets and `recipe.json`. Download/extract the
runtime package once per host, verify its manifest, and keep that directory read-only. The release
source commit and executable digest are both required; a Git tree is not presented as a commit.
Release source snapshots are created after the measured builds and verified against every frozen
source byte. Their signed commit IDs attest those sources; they are not a claim of a bit-identical
rebuild under the new commit metadata. Historical build receipts keep their original identities.
See [SIGNING.md](SIGNING.md) for the release's signing identity and verification commands.
The configuration generator writes four rank payloads and shell commands. It does not start
containers or change a proxy registration.

Example for EXL3/BF16, with a192K context limit and eight active lanes:

```sh
python3 render_config.py \
  --expert-checkpoint exl3 --precision bf16 \
  --runtime-root /srv/glm/runtime --model-root /srv/weights/glm53-exl3 \
  --drafter-root /srv/weights/glm53-dflash2 --cache-root /var/cache/glm53 \
  --master 192.168.2.201 --roce-address-range 10.78.0.0/15 \
  --context 196608 --parallel 8 --verify-rows 32 --verify-runtime \
  --prompt-cache-gib 1.75 --tier off --output /tmp/glm53-rendered
```

Choose `--precision q4`, `q4-fp8` or `q8` to render a different EXL3 option. Set `--hosts` to the four
rank hostnames if they differ from the measured fleet. Reserve distinct HTTP, NCCL bootstrap,
rank-control and gather-bootstrap ports using `--port`, `--nccl-port` and `--plan-port`.
NCCL's rank-control port is its bootstrap port plus one. Only rank0 provides the public HTTP endpoint.

The generated payloads use host networking, the CUDA device, RDMA devices, locked memory and a
native executable entrypoint. The measured interfaces are the first ports on the two HCAs, GID3,
traffic class104. Check the generated interface names and subnet against the actual switched fleet.
Reserve the designated gather CPU within each rank's cpuset. TP2/direct-link recipes are a separate
qualification and are not enabled by changing the world size in a TP4 payload.

Start followers before rank0 using the generated commands. During a healthy shutdown, terminate
rank0 first and let the followers complete their coordinated close. Runtime assets and weights stay
read-only; each rank receives its own writable cache directory. The renderer includes source,
checkpoint revision, drafter revision, weight-directory paths, precision and settings in the cache namespace.

The default qualification covers deterministic temperature0 and thinking disabled with the pinned
chat template. Changing sampling, templates, precision or checkpoint changes the request contract.
Use model/context settings within the documented memory fit. Context and parallelism share the
same UMA capacity; eight active lanes do not promise eight simultaneously full192K contexts.
The launcher accepts up to 1,048,576 tokens. TP4's completed long-context measurements reach
192K. Available memory and the active lane count determine which larger settings can start.

## RAM and NVMe prompt-cache tiers

The active attention KV state resides in memory during inference. The optional disk tier stores
prompt snapshots for later reuse. A cache hit restores the snapshot into the active state; attention
does not fetch NVMe blocks on each KV access or each generated token.

`--prompt-cache-gib` limits useful RAM prompt-cache storage. `--tier nvme` adds persistent snapshots
under the per-rank cache root, and `--learn-gib` limits their disk footprint per rank. Separate
metadata/index and staging limits bound the maintenance allocations. They are not useful cache
capacity and should not be added to the useful-cache budget as if they were prompt data.

```sh
python3 render_config.py \
  --expert-checkpoint exl3 --precision bf16 \
  --runtime-root /srv/glm/runtime --model-root /srv/weights/glm53-exl3 \
  --drafter-root /srv/weights/glm53-dflash2 --cache-root /var/cache/glm53 \
  --master 192.168.2.201 --roce-address-range 10.78.0.0/15 \
  --prompt-cache-gib 1.75 --tier nvme --learn-gib 32 --verify-runtime \
  --output /tmp/glm53-nvme-rendered
```

The conservative disk-persistence threshold is137216 tokens. `--require-cache-tokens` changes that
threshold; the renderer requires it to fit within the context limit. Use a fresh namespace when
changing source, precision, checkpoint or snapshot layout. Do not reuse older experimental cache
prefixes merely because a directory has the same model name. Cleanup should target closed,
unreferenced cache roots after verifying their ownership; weight directories are separate.

The current long-context matrix qualifies the coherent BF16 executable with disk-off, disk seed,
fresh-process disk reopen, RAM, eviction, cancellation and a real single-bit corrupted snapshot.
It verifies native token IDs and text against independently cold requests. Corruption causes a
failed-read fallback and subsequent reuse remains exact. This is finite tested fault coverage,
not a claim about every filesystem or crash mode. Other projection modes share the current worker
and receive separate cache namespaces; the long-context timing matrix itself is BF16-specific.

<!-- BEGIN CACHE MEASUREMENTS -->

### Native Tiered KV cache: release measurements

EXL3, BF16 projections/head, FP8 KV, TP4/TC104. Fresh same-binary cold, disk-seed, reopened-disk and RAM-only lifetimes. All38 normal responses match independent cold token IDs/text/completion; seven corruption-recovery responses also match.

| Case | Role | Prompt tokens | Cached tokens | Native prefill s | Native TTFT s | Full request s |
|---|---|---:|---:|---:|---:|---:|
| long160-plain | baseline | 163810 | 0 | 80.566 | 80.572 | 80.837 |
| long192-plain | baseline | 192934 | 0 | 95.886 | 95.893 | 96.150 |
| long160-plain | seed | 163810 | 0 | 81.629 | 81.638 | 81.934 |
| long192-plain | seed | 192934 | 0 | 96.090 | 100.688 | 100.965 |
| long160-plain | reopen | 163810 | 161792 | 1.050 | 2.948 | 3.198 |
| long192-plain | reopen | 192934 | 192512 | 0.330 | 2.480 | 2.734 |
| long160-plain | ram | 163810 | 0 | 80.067 | 80.075 | 80.329 |
| long192-plain | ram | 192934 | 0 | 95.301 | 95.307 | 95.560 |

Configured useful RAM cache:1.75GiB per rank. Actual accounted metadata peaked at3.26MiB per rank; its ceiling was64MiB. Serial pinned staging measured138.20MiB per rank, below the192MiB ceiling. Weights, live KV/recurrent state and useful cached snapshots are separate from maintenance metadata.

The reopened lifetime performed four snapshot restores per rank over12 completed requests plus a cancelled request. Cold seed and RAM-only arms performed zero disk restores. Restored state is then used in live memory; this is not an NVMe read for each attention/KV use.

At192K, cold disk-seed prefill was96.090s versus95.301s for RAM-only, while TTFT was100.688s versus95.307s. The additional post-prefill snapshot-credit delay remains visible. These are single sequential samples, not an isolated timing guarantee.

A real one-bit checksum failure was detected. Subsequent damaged-prefix fallback, RAM hit, other-prefix, short plain/drafted and long400-token copy checks passed; all20 private lifetimes closed normally with empty saved driver deltas. Actual write failures, cancellations and index refusals were zero.

These measurements qualify the named BF16 configuration and fresh namespace. They do not establish every filesystem fault or another precision profile’s tiered-cache behavior. Source, executable, module and archive identities are recorded in SOURCE.json and RELEASE-ASSETS.json.

<!-- END CACHE MEASUREMENTS -->

## Other controls

- `--verify-rows 32` is the measured default. `16` provides a smaller active verification window;
  it is not a claim of equivalent performance on all workloads. Neural draft depth and suffix-copy
  proposals have distinct limits.
- `--prefill stepped` is the measured default for long requests. `whole` is a separate prefill
  scheduling choice; keep the stated default when reproducing the tables.
- `--prompt-cache-gib 0 --tier off` reproduces our cache-disabled speed configuration.
- Background warm turns remain disabled. Unqualified precision/checkpoint combinations are refused.
- The borrowed gather backend is fixed for this release. Target verify graphs remain off; qualified
  FFN and drafter graphs remain on. Do not enable unrelated transport or graph flags together and
  assume the measured schedule is preserved.

Source/build receipts identify the exact compiler,146 embedded images, optional modules, patched
NCCL and borrowed native transport. Native kernel/quantizer changes require their own numerical
and model gates. The performance ledger retains rejected and neutral candidates alongside gains.
