# Actual Mia eight-lane benchmark

Run `python3 verify.py .` here for CPU-only request/client/SSE/clock verification. This is an actual TP4 run of the unchanged Mia engine and original acc.py/acc8.py: 13+28 requests, including two waves each at C4, C6 and C8. Eight serving lanes were explicitly configured. Context196608 and prompt cache0 match our speed configuration; reserve16GiB and CUDA limit100GiB are explicit memory settings. The original default four-lane result remains a separate configuration. New values are measured, not extrapolated from queued requests. This run does not repeat the full959 quality suite.

Rates use aggregate post-first tokens over the complete wave. Legacy per-stream sums are retained separately. Two repetitions with original random prompt suffixes are a finite sequential comparison, not a statistical certificate.
