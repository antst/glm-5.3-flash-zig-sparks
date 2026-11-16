test {
    _ = @import("tests/cuda/contract_exl3_whole_ref.zig");
}

test "EXL3 whole gates: fifteen compiled resources fit and the thread-limit control fails" {
    const std = @import("std");
    const registry = @import("core").contract.registry;
    var found: usize = 0;
    for (registry.all_cuda) |record| if (std.mem.eql(u8, record.launch.image, "exl3_prompt_whole")) {
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{record}, true));
        found += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), found);
    for (registry.cuda_controls) |record| if (std.mem.indexOf(u8, record.launch.name, "EXL3 whole") != null) {
        try std.testing.expect(try registry.checkCuda(std.testing.allocator, &.{record}, true) > 0);
    };
}

test "EXL3 whole gates: actual five-SM cubins equal driver-normalized records" {
    if (!@import("whole_contract_options").image_present) return error.SkipZigTest;
    const std = @import("std");
    const core = @import("core");
    const bytes = @embedFile("fatbin_exl3_prompt_whole");
    const a = std.testing.allocator;
    const images = try core.contract.cubin.images(a, bytes);
    defer core.contract.cubin.freeImages(a, images);
    try std.testing.expectEqual(@as(usize, 5), images.len);
    var count: usize = 0;
    for (images) |image| {
        const at = for (core.contract.limits.sms, 0..) |sm, i| {
            if (sm.sm == image.sm) break i;
        } else return error.UnknownCompiledSm;
        const kernels = try core.contract.cubin.kernels(a, image.elf, image.sm);
        defer a.free(kernels);
        try std.testing.expectEqual(@as(usize, 3), kernels.len);
        for (kernels) |kernel| {
            const record = for (core.contract.registry.all_cuda) |record| {
                if (std.mem.eql(u8, record.launch.image, "exl3_prompt_whole") and
                    std.mem.eql(u8, record.launch.symbol, kernel.name)) break record;
            } else return error.UnrecordedCompiledKernel;
            try std.testing.expectEqual(record.usage[at], kernel.usage);
            count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 15), count);
}
