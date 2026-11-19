//! A separate drafter's identity belongs to its persisted rings, once per opened store.
const std = @import("std");

pub fn withDrafter(base: [32]u8, drafter: ?[32]u8, bits: u8) [32]u8 {
    const artifact = drafter orelse return base;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("glm-native-tier/external-drafter-v1");
    hash.update(&base);
    hash.update(&artifact);
    hash.update(&.{bits});
    return hash.finalResult();
}

test "native tier identity: no external drafter preserves base, rings distinguish artifact and bits" {
    const base: [32]u8 = @splat(0x11);
    const draft: [32]u8 = @splat(0x22);
    try std.testing.expectEqualSlices(u8, &base, &withDrafter(base, null, 4));
    const selected = withDrafter(base, draft, 4);
    try std.testing.expectEqualStrings("5d7c51c83305389d56b3f1238140d0aefd6290ae31ad47c91806fe7d570bfff0", &std.fmt.bytesToHex(selected, .lower));
    try std.testing.expectEqualSlices(u8, &selected, &withDrafter(base, draft, 4));
    var different = draft;
    different[17] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &selected, &withDrafter(base, different, 4)));
    try std.testing.expect(!std.mem.eql(u8, &selected, &withDrafter(base, draft, 0)));
    var other_base = base;
    other_base[3] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &selected, &withDrafter(other_base, draft, 4)));
}
