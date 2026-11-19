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

The measured limits are 196608 context tokens and eight active lanes. TP2 uses stepped
prefill and 32 verification rows. The attention KV and recurrent state share the host's
UMA capacity.

## RAM and NVMe

`--prompt-cache-gib` controls the RAM prompt snapshot budget. The Q8/BF16 runtime has a
measured optional NVMe tier. Set `--tier nvme --prompt-cache-gib 1.75 --learn-gib 32`
to render it with a distinct per-rank cache namespace. Cold inference keeps active KV
in memory. A disk hit restores a snapshot into memory for subsequent attention.

The Q8 validation covered 160K/192K requests, cache reopening and checksum-failure fallback.
The current TP2 BF16 and Q4 profiles are provided with `--tier off`. See the
[TP4 cache documentation](TP4.md#ram-and-nvme-prompt-cache-tiers) for the shared cache model.

## Quality and performance

The tables use the TP4 format. Mia's original Zig clients provide the C1, C4, C6 and C8
rates and the long-prompt diagnostics. The balanced benchmark holds 400 completion tokens
per request, uses two repetitions and reports throughput over a common client-clock window.
Speed runs use eight active lanes, FP8 KV, NVMe off and RAM prompt cache off.

<!-- BEGIN TP2 MEASUREMENTS -->
<!-- END TP2 MEASUREMENTS -->
