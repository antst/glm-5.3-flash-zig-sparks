//! GLM contract fixtures and loader controls that run without a GPU or kernel images.
const std = @import("std");

pub fn host(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const audit = b.addRunArtifact(b.addExecutable(.{ .name = "glm-dflash-assets-host", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_dflash_assets_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) }));
    audit.addPassthruArgs();
    b.step("glm-dflash-assets-host", "Read real cubin resource records and check the private TP4 launches on the CPU")
        .dependOn(&audit.step);
    const tp4 = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_dflash_tp4_contract_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }), .filters = &.{"DFlash TP4 contract:"} }));
    b.step("test-glm-dflash-tp4-contract-host", "Own-head layout controls and semantic device command without a GPU")
        .dependOn(&tp4.step);
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_contract_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    const run = b.addRunArtifact(tests);
    const step = b.step("test-glm-contract-host", "GLM contract fixtures, frozen exports and clean-before-tamper controls");
    step.dependOn(&run.step);
    all.dependOn(step);
}
