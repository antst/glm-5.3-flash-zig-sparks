//! Existing GLM family recording controls on the CPU runtime.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    b.step("test-glm-warm-turns-host", "Strict background warm-turn opt-in and rank agreement")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM warm turns:"} })).step);
    b.step("test-glm-nvfp4-fact-host", "NVFP4 decode opt-in and actual grouped launch records")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM NVFP4 fact:"} })).step);
    b.step("test-glm-latent-mma-host", "Prompt tensor-core routing, immutable ownership and rank agreement")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"latent MMA:"} })).step);
    // GLM CUDA's startup watcher (named phases, a warning while NCCL init or a load blocks), from cuda.zig's room
    b.step("test-startup-watch-host", "Blocked startup phase warnings without a driver or communicator").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"startup watcher:"} })).step);
    b.step("test-glm-affine-target-host", "Affine Q4 target settings, storage and actual launch records")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"affine target:"} })).step);
    b.step("test-glm-head-fp8-host", "Explicit Q4 FP8 head scope, geometry and scalar quantizer reference")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"head FP8:"} })).step);
    b.step("test-glm-decode-select-host", "Multi-CTA decode selection bounds and startup scope")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"decode select:"} })).step);
    const host = b.graph.host;
    const target_q4 = b.createModule(.{ .root_source_file = b.path("zig/glm_target_q4_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "glm_cuda", .module = glm } } });
    b.step("test-glm-target-q4-host", "Target Q4 references, graph entrypoint and live-resource declarations")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = target_q4, .filters = &.{"target Q4"} })).step);
    b.step("test-glm-q4-optimized-host", "Optional target Q4 scopes with corrected drafting and optimized startup")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"optional Q4 optimized:"} })).step);
    b.step("test-glm-kda-adaptive-host", "KDA threshold, physical16 owner and original launch records without devices")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"KDA "} })).step);
    b.step("test-glm-admission-wait-host", "Strict GLM idle admission settings and rank agreement")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM admission wait:"} })).step);
    b.step("test-glm-ffn-graphs-host", "Compute-only FFN graph ownership and bounded eager fallback")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"FFN graphs:"} })).step);
    const row16_contract = b.createModule(.{ .root_source_file = b.path("zig/glm_row16_contract_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "glm_cuda", .module = glm } } });
    b.step("test-glm-row16-contract-host", "Actual State/attention/EXL3 device entrypoint semantic compilation").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = row16_contract })).step);
    b.step("test-glm-q4-wide-host", "Explicit Q4 fp32 symbol and recorded launch ABI without a device")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM Q4 wide:"} })).step);
    const q4_contract = b.createModule(.{
        .root_source_file = b.path("zig/glm_q4_wide_contract_host.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "glm_cuda", .module = glm } },
    });
    b.step("test-glm-q4-wide-contract-host", "Q4 fp32 contract inventory and semantic checks without a device")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = q4_contract, .filters = &.{"GLM Q4 wide contract:"} })).step);
    const warm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_warm_peers_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "heat", .module = mods.heat } } });
    const warm_run = b.addRunArtifact(b.addTest(.{ .root_module = warm, .filters = &.{"GLM warm peers:"} }));
    b.step("test-glm-warm-peers-host", "Selected GLM warm-up peers and NCCL posting failures without devices").dependOn(&warm_run.step);
    step.dependOn(&warm_run.step);
    const startup = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_start_test.zig"), .target = host, .optimize = .debug });
    const startup_run = b.addRunArtifact(b.addTest(.{ .root_module = startup }));
    b.step("test-glm-startup-host", "Page refusals propagate through rank liveness before engine construction").dependOn(&startup_run.step);
    step.dependOn(&startup_run.step);
    b.step("test-glm-tp3-host", "Uneven KDA head shares through actual projection and load planning, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM TP3:"} })).step);
    // TF_GLM_INDEX_CUDA: the launch rules, and (its own slower step) glm_index_*.cuh as host C++ against plain models
    const index_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM index:"} }));
    b.step("test-glm-index-host", "GLM's prompt indexer launch rules without devices").dependOn(&index_run.step);
    step.dependOn(&index_run.step);
    const index_emulate = b.addExecutable(.{ .name = "glm-index-emulate", .root_module = b.createModule(.{ .target = host, .optimize = .ReleaseSafe, .link_libcpp = true }) });
    index_emulate.root_module.addCSourceFile(.{ .file = b.path("tools/glm_index_emulate/emulate.cpp"), .flags = &.{ "-std=c++20", "-O1" } });
    index_emulate.root_module.addIncludePath(b.path("tools/glm_index_emulate"));
    const index_emulated = b.addRunArtifact(index_emulate);
    index_emulated.expectExitCode(0);
    b.step("test-glm-index-emulate", "GLM's prompt indexer kernels as host C++ against a plain top-K and a model of tf_glm_scores").dependOn(&index_emulated.step);
    const kv8_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM FP8 KV:"} }));
    b.step("test-glm-fp8-kv-host", "GLM latent cache layout, mode agreement and recorded CUDA launches without devices").dependOn(&kv8_run.step);
    step.dependOn(&kv8_run.step);
    const verify_rows = b.createModule(.{ .root_source_file = b.path("zig/glm_verify_rows_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = mods.lanes } } });
    b.step("test-glm-verify-rows-host", "Physical32 State offsets, active8/16/32 admission and partial commits without devices")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = verify_rows, .filters = &.{"verify rows:"} })).step);
    const retention = b.createModule(.{ .root_source_file = b.path("zig/glm_retention_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda } } });
    b.step("test-glm-retention-host", "Actual sequence bind/release/reset/free ownership without a device")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = retention, .filters = &.{"sequence retention:"} })).step);
    const profile_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM prompt profile:"} }));
    b.step("test-glm-prompt-profile-host", "Paired prompt-stage event timing, overlap, failure and disabled controls without devices").dependOn(&profile_run.step);
    step.dependOn(&profile_run.step);
    const pages = b.createModule(.{ .root_source_file = b.path("zig/glm_pages_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "core", .module = mods.core } } });
    const pages_run = b.addRunArtifact(b.addTest(.{ .root_module = pages, .filters = &.{"GLM pages:"} }));
    b.step("test-glm-pages-host", "Actual GLM page ownership, refusal and rollback without devices").dependOn(&pages_run.step);
    step.dependOn(&pages_run.step);
    const api = glm.import_table.get("engine_api").?; // the family's own (the shared store's types): one module, one set of types
    const adapter = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_prompt_pages.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = mods.core }, .{ .name = "engine_api", .module = api } } });
    const image_cache = b.createModule(.{ .root_source_file = b.path("zig/glm_image_cache_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "core", .module = mods.core }, .{ .name = "prompt_pages", .module = adapter } } });
    const image_cache_run = b.addRunArtifact(b.addTest(.{ .root_module = image_cache }));
    b.step("test-glm-image-cache-host", "Image-content prefix reuse over the shared store and suffix rows across both ranks without devices").dependOn(&image_cache_run.step);
    step.dependOn(&image_cache_run.step);
    const page_contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_glm_controls_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    const page_contract_run = b.addRunArtifact(b.addTest(.{ .root_module = page_contract, .filters = &.{"GLM page contract:"} }));
    b.step("test-glm-page-contract-host", "GLM paged contract case and planted-control inventory without devices").dependOn(&page_contract_run.step);
    step.dependOn(&page_contract_run.step);
    const admission = b.createModule(.{ .root_source_file = b.path("zig/glm_admission_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "engine_api", .module = api } } });
    const admission_run = b.addRunArtifact(b.addTest(.{ .root_module = admission, .filters = &.{"GLM CUDA admission:"} }));
    b.step("test-glm-admission-host", "Real GLM admission capabilities through the lane worker without GPU work").dependOn(&admission_run.step);
    step.dependOn(&admission_run.step);
    const copies = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_snaps.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "core", .module = mods.core } } });
    const snapshots = b.createModule(.{ .root_source_file = b.path("zig/glm_snapshot_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "engine_api", .module = api }, .{ .name = "copies", .module = copies } } });
    const live = b.createModule(.{ .root_source_file = b.path("zig/glm_live_state_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    const draft_head = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_head_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-dflash-head-host", "DFlash-only head policy, ownership and actual State allocation without devices")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = draft_head, .filters = &.{ "GLM DFlash head:", "MTP widened head" } })).step);
    b.step("test-glm-live-state-host", "Actual compact live-bank images, saved parity and snapshot ownership without devices")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = live, .filters = &.{ "GLM live state:", "GLM disk image:" } })).step);
    b.step("test-glm-disk-live-host", "Compact and legacy disk image byte layouts through the GLM module itself")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{ "GLM live state:", "GLM disk image:" } })).step);
    const snapshots_run = b.addRunArtifact(b.addTest(.{ .root_module = snapshots }));
    b.step("test-glm-snapshots-host", "Prompt-sized GLM state copies, page budgets and rank ordering without devices").dependOn(&snapshots_run.step);
    const nvme = b.createModule(.{ .root_source_file = b.path("zig/glm_nvme_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-nvme-host", "Actual GLM image and bounded native NVMe store, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = nvme, .filters = &.{"GLM NVMe:"} })).step);
    const progress = b.createModule(.{
        .root_source_file = b.path("zig/glm_nvme_progress_host.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "glm", .module = glm }},
    });
    b.step("test-glm-nvme-progress-host", "Real NVMe work feedback and bounded pressure watchdog")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = progress, .filters = &.{"GLM NVMe progress:"} })).step);
    const compact = b.createModule(.{ .root_source_file = b.path("zig/glm_nvme_compact_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-nvme-compact-host", "Native tier shared ownership, metadata and admission without devices")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = compact, .filters = &.{"GLM compact:"} })).step);
    const store_tests = b.createModule(.{ .root_source_file = b.path("zig/glm_nvme1107_store_host.zig"), .target = host, .optimize = .debug, .link_libc = true });
    b.step("test-glm-nvme-store-host", "Actual spill workers, crash controls and recovery without devices")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = store_tests })).step);
    const metadata = b.createModule(.{ .root_source_file = b.path("zig/prompt_token_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = api }} });
    b.step("test-prompt-metadata-host", "Shared canonical histories through the actual prompt cache")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = metadata })).step);
    b.step("test-lane-failure-host", "Actual failed-round ownership, interrupted prompt and later-request recovery")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{ "failed round", "mid-fill", "failed piece", "piece fails", "paused peer", "round planner" } })).step);
    b.step("test-prompt-cache-core-host", "Full engine API, lane host and prompt-cache host regressions")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    b.step("test-prompt-cache-filtered-host", "Prompt, snapshot, tier and cache regressions")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{ "prompt", "snapshot index", "cache", "rows:", "tier:" } })).step);
    step.dependOn(&snapshots_run.step);
    b.step("test-glm-prompt-settings-host", "Actual opt-in settings and rank agreement on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"prompt K experiment"} })).step);
    b.step("test-glm-greedy-batch-host", "Greedy verification record batching without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"greedy batching:"} })).step);
    b.step("test-glm-dense-segments-host", "Dense shared MLA table and scratch checks without a GPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"dense segment"} })).step);
    const latent_contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_latent_mma_contract.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "glm_cuda", .module = glm } } });
    b.step("test-glm-latent-mma-contract-host", "Independent periodic FP64 oracle and reduced-precision controls")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = latent_contract, .filters = &.{"latent MMA"} })).step);
    const dense_contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_dense_segmented_contract.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "glm_cuda", .module = glm } } });
    b.step("test-glm-dense-contract-host", "Dense latent replacement scatter and actual launch records on CPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dense_contract, .filters = &.{"dense segmented"} })).step);
    b.step("test-glm-copy-75c-host", "GLM copy selection against frozen Python fixtures and request ownership")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM 75c copy:"} })).step);
    b.step("test-glm-dflash-policy-host", "Actual GLM facts, greedy budgets and confidence allocation on CPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"DFlash policy:"} })).step);
    b.step("test-glm-dflash-edge-host", "Actual DFlash selector scores, chain chances and startup agreement on the CPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"DFlash edge:"} })).step);
    b.step("test-glm-dflash-graphs-host", "DFlash compute graph ownership and eager fallback without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"DFlash graphs:"} })).step);
    b.step("test-glm-verify-graphs-host", "Verify graph keys and coordinated replay without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM verify graphs:"} })).step);
    b.step("test-glm-sample-batch-host", "Sampled padded-gather bytes and ownership without a GPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"sample batching:"} })).step);
    b.step("test-glm-dflash-attention-host", "TP4 draft attention views and stage ordering without a GPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"DFlash attention:"} })).step);
    b.step("test-glm-dflash-ingest-graphs-host", "Context ingest graph ownership and shape bounds without a GPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"DFlash ingest graphs:"} })).step);
    const attention_load = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_attention_load_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda } } });
    b.step("test-glm-dflash-attention-load-host", "Actual TP4 head symbols and missing-asset refusal without devices")
        .dependOn(&b.addRunArtifact(b.addExecutable(.{ .name = "glm-attention-assets-host", .root_module = attention_load })).step);
    const mm_record = b.createModule(.{ .root_source_file = b.path("zig/test_support/glm_mm_cuda.zig"), .target = host, .optimize = .debug });
    const mm_actual = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_mm.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = mm_record }} });
    const mm_policy = b.createModule(.{ .root_source_file = b.path("zig/glm_mm_policy_host.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = mm_record }, .{ .name = "mm", .module = mm_actual } } });
    b.step("test-glm-mm-policy-host", "Actual prompt K policy dispatch without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mm_policy })).step);
    const drafter_host = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-dflash-host", "DFlash rank buffers, partial loader ownership and pending records on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = drafter_host, .filters = &.{"glm dflash:"} })).step);
    const packed_host = b.createModule(.{ .root_source_file = b.path("zig/glm_packed_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-packed-host", "Packed GLM launch records, cap parsing and native loader on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = packed_host, .filters = &.{"glm packed:"} })).step);
    const ring_host = b.createModule(.{ .root_source_file = b.path("zig/glm_ring_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    b.step("test-glm-ring-host", "Actual GLM MTP rollback pool through recording CUDA copies").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ring_host, .filters = &.{"glm ring:"} })).step);
    const taps_tp = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM prompt taps:"} }));
    b.step("test-glm-prompt-taps-host", "DFlash2 prompt taps hold every row at TP>1 through recording copies and emulated NCCL").dependOn(&taps_tp.step);
    step.dependOn(&taps_tp.step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    const whole_host = b.createModule(.{ .root_source_file = b.path("zig/exl3_whole_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    b.step("test-exl3-whole-host", "Whole-prompt direct slots, scratch and agreed settings without a GPU")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = whole_host, .filters = &.{"whole prompt:"} })).step);
    const whole_contract = b.createModule(.{ .root_source_file = b.path("zig/exl3_whole_contract_host.zig"), .target = host, .optimize = .fast, .link_libc = true, .imports = &.{.{ .name = "core", .module = mods.core }} });
    const whole_options = b.addOptions();
    const whole_image = b.option([]const u8, "whole-contract-fatbin", "Exact optional whole-prompt image for CPU resource audit");
    whole_options.addOption(bool, "image_present", whole_image != null);
    whole_contract.addOptions("whole_contract_options", whole_options);
    if (whole_image) |path| whole_contract.addAnonymousImport("fatbin_exl3_prompt_whole", .{ .root_source_file = .{ .cwd_relative = path } });
    b.step("test-exl3-whole-contract-host", "Independent whole-prompt FP64 references and accumulator sensitivity")
        .dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = whole_contract, .filters = &.{"EXL3 whole gates:"} })).step);
    const prompt_host = b.createModule(.{ .root_source_file = b.path("zig/exl3_prompt_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes } } });
    const prompt_run = b.addRunArtifact(b.addTest(.{ .root_module = prompt_host, .filters = &.{"exl3 prompt:"} }));
    prompt_run.has_side_effects = true;
    b.step("test-exl3-prompt-host", "Actual common prompt dispatch and scratch with recording CUDA calls").dependOn(&prompt_run.step);
}

pub fn compile(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, comptime makeGlm: anytype) void {
    const fp8_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_ref_fp8.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    const fp8_object = b.addObject(.{ .name = "glm-fp8-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_fp8_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = fp8_gate }} }) });
    _ = fp8_object.getEmittedBin();
    b.step("compile-glm-fp8-gate-host", "Compile synthetic FP8 write/read and FP64 device checks with inert images, never execute").dependOn(&fp8_object.step);
    const glm_open = b.addObject(.{ .name = "glm-open-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_open_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) }} }) });
    _ = glm_open.getEmittedBin();
    b.step("compile-glm-native-open", "Compile the actual GLM Linux open body with empty GPU images").dependOn(&glm_open.step);
    const prompt_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_mm_fixed_tests.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) }, .{ .name = "cuda", .module = cuda } } });
    const prompt_gate_compile = b.addObject(.{ .name = "glm-prompt-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = prompt_gate }} }) });
    _ = prompt_gate_compile.getEmittedBin();
    b.step("compile-glm-prompt-gate-host", "Compile the actual device prompt comparison gate with inert images").dependOn(&prompt_gate_compile.step);
}
