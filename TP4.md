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

## Quality and performance

Our TP4 speed measurements use TensorFold `c829c30e`. The Mia row uses its original Zig engine `80be4d68`. The decode runs use the switched setup at traffic class 104, an eight-request limit, 196,608-token context, FP8 KV, and both prompt-cache tiers off. The quality tables identify the revisions used for scoring.

### Original Mia Zig benchmark

C1 shows per-stream decode rates. C4, C6, and C8 show whole-wave throughput. All rates are in tok/s.

| Profile | C1 prose | C1 code | C4 prose | C6 prose | C8 prose | Quality /959 | Small code / structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 54.37 | 77.91 | 128.85 | 153.51 | 169.99 | 881 | 8/8 · 5/8 |
| Ours Q4/BF16 | 76.93 | 91.99 | 154.19 | 178.31 | 197.27 | 878 | 8/8 · 7/8 |
| Ours Q4/FP8 | 80.17 | 96.13 | 152.11 | 183.88 | 200.71 | 882 | 8/8 · 7/8 |
| Ours Q8/BF16 | 72.60 | 93.75 | 144.95 | 168.43 | 192.79 | 878 | 8/8 · 5/8 |
| Mia Zig Q8/BF16 | 63.54 | 85.16 | 129.69 | 162.56 | 175.63 | 878 | 8/8 · 5/8 |

### Balanced benchmark

The same fixed prompts run twice at each concurrency, with 400 completion tokens per request. Rates use the common client-clock window.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 70.38 | 120.59 | 159.48 |
| Ours Q4/BF16 | 93.09 | 138.66 | 173.88 |
| Ours Q4/FP8 | 96.08 | 143.22 | 178.57 |
| Ours Q8/BF16 | 90.73 | 141.60 | 182.10 |
| Mia Zig Q8/BF16 | 80.40 | 128.70 | 166.24 |

### Whole-wave throughput

| Profile | C4 tok/s | C6 tok/s | C8 tok/s |
| --- | ---: | ---: | ---: |
| Ours BF16/BF16 | 128.85 | 153.51 | 169.99 |
| Ours Q4/BF16 | 154.19 | 178.31 | 197.27 |
| Ours Q4/FP8 | 152.11 | 183.88 | 200.71 |
| Ours Q8/BF16 | 144.95 | 168.43 | 192.79 |
| Mia Zig Q8/BF16 | 129.69 | 162.56 | 175.63 |

### Quality by category

Each row uses the same 959-case inventory.

| Profile | Engine | Full /959 | qa /150 | reason /40 | arith /40 | track /40 | gsm8k /200 | mmlu /285 | follow /40 | legacy_humaneval_chat /164 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | `1e3ba2cf` | 881 | 129 | 40 | 40 | 40 | 195 | 253 | 28 | 156 |
| Ours Q4/BF16 | `1e3ba2cf` | 878 | 127 | 40 | 40 | 39 | 195 | 250 | 33 | 154 |
| Ours Q4/FP8 | `1e3ba2cf` | 882 | 128 | 40 | 40 | 39 | 195 | 252 | 35 | 153 |
| Ours Q8/BF16 | `1e3ba2cf` | 878 | 131 | 40 | 39 | 40 | 196 | 249 | 29 | 154 |
| Mia Zig Q8/BF16 | `80be4d68` | 878 | 131 | 40 | 39 | 40 | 196 | 249 | 29 | 154 |

### Original Mia bench2 long prompts

Each cell uses the measured prompt length and client time to first token. The client generates a different prompt per length.

Mia’s recorded bench2 run used a four-request limit and 8 GiB of RAM prompt cache. These workloads used up to four requests and logged zero cached tokens.

| Profile | 16K tokens | TTFT s | Prompt tok/s | 64K tokens | TTFT s | Prompt tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 15882 | 7.909 | 2008.2 | 63533 | 31.101 | 2042.8 |
| Ours Q4/BF16 | 16041 | 8.052 | 1992.2 | 64361 | 31.819 | 2022.7 |
| Ours Q4/FP8 | 16093 | 8.098 | 1987.4 | 64369 | 31.871 | 2019.6 |
| Ours Q8/BF16 | 16026 | 7.973 | 2010.0 | 64347 | 31.537 | 2040.4 |
| Mia Zig Q8/BF16 | 16046 | 8.980 | 1786.9 | 64596 | 33.867 | 1907.4 |

### Original Mia bench2 mixed and paired requests

A roughly 50K-token prefill runs beside one or three decoders. Gaps are the worst per-stream values. The paired prompts are roughly 30K tokens each.

| Profile | 50K +1 decoder TTFT s | 50K +3 decoders TTFT s | Worst p95 gap ms | Worst maximum gap ms | 30K alone TTFT s | 30K pair TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours BF16/BF16 | 36.61 | 36.83 | 1054 | 1286 | 14.93 | 29.93 / 15.76 |
| Ours Q4/BF16 | 36.89 | 37.40 | 1057 | 1314 | 15.14 | 16.06 / 30.36 |
| Ours Q4/FP8 | 37.03 | 37.77 | 1070 | 1302 | 15.20 | 17.40 / 30.55 |
| Ours Q8/BF16 | 36.71 | 37.07 | 1054 | 1340 | 15.09 | 17.20 / 30.39 |
| Mia Zig Q8/BF16 | 44.07 | 44.37 | 77 | 2610 | 16.39 | 33.56 / 16.38 |

[Measurement record](measurements.json)

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
