//! Verify graphs retain direct state pointers; every invocation coordinates the hit decision before NCCL replay.
const std = @import("std");
const cuda = @import("cuda");
const gc = cuda.graph_cache;
const dsa = @import("cuda_dsa_block.zig");

pub const max_parts = 16;
pub const max_dsa = 16;

pub const Geometry = struct {
    cache: dsa.Caches = .{ .lc = 0, .ik = 0, .ig = 0, .pk = 0, .pk_rows = 0 },
    dense: bool = false,
    sparse: bool = false,
    chunks: usize = 0,
    pools: usize = 0,
    score_blocks: usize = 0,
};

pub const Part = struct {
    state: usize = 0,
    arena: u64 = 0,
    pos_device: u64 = 0,
    base: usize = 0,
    row0: usize = 0,
    rows: usize = 0,
    parity: u64 = 0,
    attention: [max_dsa]Geometry = @splat(.{}),
};

pub const Key = struct {
    scratch: u64,
    rows: usize,
    count: usize,
    parts: [max_parts]Part = @splat(.{}),
};

/// All suballocations, kernel modules, precision and communicator settings stay immutable under these owners.
pub const Binding = struct {
    engine: usize,
    weights: usize,
    modules: usize,
    stream: usize,
    comm: usize,
    taps: u64,
    tap_function: usize,
};

pub const Cache = gc.Cache(Key);

pub const Graphs = struct {
    driver: gc.CudaEngine,
    cache: Cache,
    binding: ?Binding = null,
    generation: u64 = 0,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .driver = .{ .d = d }, .cache = undefined };
        const settings: gc.Settings = .{ .max = 32, .min_keep = 0, .hold = true };
        g.cache = Cache.init(gpa, g.driver.engine(), settings);
        g.cache.settings = settings; // the effective enable bit was agreed at startup, not read again per rank
        g.cache.name = "glm-verify";
        return g;
    }

    pub fn destroy(g: *Graphs, gpa: std.mem.Allocator) void {
        g.cache.deinit();
        gpa.destroy(g);
    }

    /// The caller drains before state restore/free, and destroys graphs before their direct pointer owners.
    pub fn invalidate(g: *Graphs) void {
        g.cache.dropAll();
        g.generation +%= 1;
    }

    pub fn run(g: *Graphs, key: Key, binding: Binding, s: cuda.Stream, vote: gc.Agree, body: gc.Body) !gc.Outcome {
        if (g.binding) |old| if (!std.meta.eql(old, binding)) return error.VerifyGraphOwnerChanged;
        g.binding = binding;
        g.cache.agree = vote;
        if (g.cache.settings.on and !g.cache.failed) {
            // A local eviction or allocation failure must never leave another rank launching a different schedule.
            if (!try vote.all(vote.ctx, g.cache.has(key))) g.cache.forget(key);
        }
        return g.cache.run(key, s, g.generation, body);
    }
};
