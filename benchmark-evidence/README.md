# Original Mia Zig benchmark evidence

Run `python3 verify.py .` from this directory. No server, GPU or packages are needed.

Each profile contains all 63 recorded HTTP requests, actual monotonic line-arrival clocks, and byte-exact SSE responses from the unchanged acc.py, acc8.py and bench2.py clients. Original EVIDENCE files are retained; working copies change only raw_path to a relative filename. The verifier checks client hashes, request bodies, all SSE hashes/usage and both C4/C6/C8 metrics. The warmup request is preserved and excluded from speed aggregates.

Reported stream-rate sums and whole-wave throughput are different metrics from the same requests. Mia has four active lanes; ours has eight. This is a finite whole-configuration comparison, not an isolated kernel or transport attribution. Prompt nonces and natural stopping are preserved. INDEX.json records whether every main row uses the final coherent build; a false value means a previous qualified row is still present and this export is not the final release evidence.
