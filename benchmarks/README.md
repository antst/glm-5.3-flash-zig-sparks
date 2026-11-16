# Reproducing the client measurements

Use Python 3.12. The clients use the standard library. `upstream/` retains the original
TensorFold HTTP benchmark sources; `mia/` retains Mia's three original harness files.
Their original repository commits and file hashes are recorded beside them. These
are clients, and do not install or replace either server's engine.

For the published speed settings, render serving with `--tier off --prompt-cache-gib 0`.
Keep the same checkpoint, precision, context, parallel count and verification width
as the table being reproduced. The client checks zero cached tokens where the server
reports them; that check does not independently attest that its NVMe writer is disabled.
Model startup is excluded. The clients retain their original warmup behavior.

## Fixed balanced matrix

```sh
python3 benchmarks/run_balanced.py \
  --url http://node-a:18888 --model GLM-5.3-Flash \
  --output /tmp/glm53-balanced
```

This runs eight distinct tasks individually, two four-task cohorts and one eight-task
cohort. Each case has two repetitions and 400 generated tokens per request: 48 timed
requests overall. Prompt files and their hashes are fixed. The copied client sources
are byte-identical to those used for the release measurements. The new portable runner
supplies their URL and file paths and reduces the saved results; it adds no sampling
fields to the client requests.

For each repetition the aggregate rate is `streams * 399 / (latest last content arrival
- earliest first content arrival)`. Take the median of the two repetitions, then the
geometric mean of case rates at each concurrency. The first token is excluded from
decode throughput. End-to-end throughput is recorded separately. The runner also
supports `--reduce-only --output EXISTING_DIRECTORY`; it refuses to overwrite an
existing summary.

The published reductions additionally join the saved raw HTTP responses with their
native token hashes, text and usage. Their transparent recorders buffered evidence
in memory and wrote files after timing. The portable direct runner retains the
underlying client intervals and token evidence, without claiming a new raw-SSE audit.

## Original TensorFold HTTP clients

```sh
python3 benchmarks/upstream/bench_openai.py http://node-a:18888 GLM-5.3-Flash \
  --tokens 512 --reps 5 --temperatures 0 --output /tmp/glm53-single512.json

python3 benchmarks/upstream/bench_concurrent.py http://node-a:18888 GLM-5.3-Flash \
  --tokens 256 --reps 3 --temperatures 0 --levels 1,4,8 --alone --serial \
  --output /tmp/glm53-concurrent256.json
```

The solo benchmark contains a raw Fibonacci completion and a chat prompt. The
concurrent benchmark contains code and chat cells; its `--alone` and `--serial`
controls compare concurrent, solo and target-only outputs. These homogeneous
concurrent prompts can share expert routes. Their rates are a separate table from
the distinct-task balanced workload.

## Mia's original clients

Run these unchanged clients against each separately configured server:

```sh
python3 benchmarks/mia/run_recorded.py acc.py http://node-a:18888 GLM-5.3-Flash run \
  --output /tmp/glm53-mia-acc
python3 benchmarks/mia/run_recorded.py acc8.py http://node-a:18888 GLM-5.3-Flash run \
  --output /tmp/glm53-mia-acc8
python3 benchmarks/mia/run_recorded.py bench2.py http://node-a:18888 GLM-5.3-Flash run \
  --output /tmp/glm53-mia-bench2
```

`run_recorded.py` preserves the original request bodies and options. It records
response bytes and arrival times in memory, then writes evidence after the client
exits. It expects 13, 28 and 22 requests respectively. Natural stopping, random
prompt suffixes and original token caps remain enabled. Concurrent rates reported
by these scripts sum individual stream rates; they do not use the balanced
common-window denominator. Random long prompts are not identical token inputs
between sequential runs.

Thermal state, clocks, cache state and run order can affect comparisons. The release
keeps measured per-profile thermal observations and finite repetition counts. Small
sequential differences are descriptive; no fixed-clock or universal thermal result
is implied.

The fixed quality-suite composition, task counts, token caps, thinking settings and
scorer hashes are recorded in [quality-protocol.json](quality-protocol.json). Those
quality scores are separate from each client's throughput and request-validation checks.
