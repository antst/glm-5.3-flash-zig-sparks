# Optional DFlash head-only load

`TF_GLM_DFLASH_ONLY_HEAD=1` loads the existing 4-bit head borrowed by external DFlash, while omitting the
unused target MTP layer, its sequence cache and its window buffer. It requires `--drafter` and global
drafting enabled. The initial scope excludes paged KV, host KV and vision; unsupported combinations
are refused before weight allocation. The setting is cached and agreed by all ranks at startup.

The default is `0`: legacy target MTP loading and allocations remain. This option does not change the
external drafter's arithmetic, quantizer, GPU kernels, graph stages or communication. Normal target
projections and head precision still follow the chosen serving profile.

A fresh cache namespace is required; cross-binary cache compatibility is not claimed. Startup logs
report the selected mode, target MTP presence and borrowed-head presence. CPU sizing checks are not
actual device memory measurements. Model parity, memory and final performance/quality results must
be recorded separately before recommending this option for serving.
