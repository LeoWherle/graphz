const std = @import("std");

/// A static (compile-time) storage for up to `Capacity` items.
pub fn StaticStorage(comptime T: type, comptime Capacity: usize) type {
    return struct {
        pub const Self = @This();
        buffer: [Capacity]T,
        count: usize,

        /// Initialize the static storage. Errors at compile-time if `capacity > Capacity`.
        pub fn init(_: ?*std.mem.Allocator, capacity: usize) Self {
            if (capacity > Capacity) {
                @compileError("Requested capacity exceeds static storage capacity");
            }
            return Self{ .buffer = undefined, .count = 0 };
        }

        pub fn deinit(_: *Self) void {}

        pub fn ensureTotalCapacity(_: *Self, capacity: usize) std.mem.Allocator.Error!void {
            if (capacity > Capacity) {
                return error.OutOfMemory;
            }
        }

        pub fn append(self: *Self, value: T) !void {
            if (self.count >= Capacity) {
                return error.OutOfMemory;
            }
            self.buffer[self.count] = value;
            self.count += 1;
        }

        /// Swap-remove an element at `index`, returning it.
        pub fn swapRemove(self: *Self, index: usize) T {
            std.debug.assert(index < self.count);
            const last_index = self.count - 1;
            const removed = self.buffer[index];
            if (index != last_index) {
                self.buffer[index] = self.buffer[last_index];
            }
            self.count -= 1;
            return removed;
        }

        pub fn get(self: *Self, index: usize) *T {
            // Caller checks bounds
            return &self.buffer[index];
        }

        pub inline fn len(self: *Self) usize {
            return self.count;
        }

        pub inline fn raw(self: *Self) []T {
            return self.buffer[0..self.count];
        }

        pub inline fn isComptime() bool {
            // We treat this storage as "compile-time" capable
            return true;
        }

        /// Optional: not used in the example unless you add advanced usage
        pub fn resetAllocator(_: *Self, _: *std.mem.Allocator) void {
            // No-op for static storage
        }

        /// Optional: demonstrate a "clone" if needed.
        /// For static storage, we can just copy contents.
        pub fn clone(self: *Self) !Self {
            var new_self = Self{ .buffer = undefined, .count = self.count };
            @memcpy(new_self.buffer[0..self.count], self.buffer[0..self.count]);
            return new_self;
        }
    };
}

/// A dynamic (runtime) storage backed by `std.ArrayList(T)`.
pub fn DynamicStorage(comptime T: type) type {
    return struct {
        const Self = @This();
        list: std.ArrayList(T),

        pub fn init(allocator: ?*std.mem.Allocator, capacity: usize) Self {
            if (allocator) |alloc| {
                var storage = Self{ .list = std.ArrayList(T).init(alloc.*) };
                storage.list.ensureTotalCapacity(capacity) catch {};
                return storage;
            }
            @compileError("Allocator is required for DynamicStorage at run-time");
        }

        pub fn deinit(self: *Self) void {
            self.list.deinit();
        }

        pub fn ensureTotalCapacity(self: *Self, capacity: usize) !void {
            try self.list.ensureTotalCapacity(capacity);
        }

        pub fn append(self: *Self, value: T) !void {
            try self.list.append(value);
        }

        /// Swap-remove an element at `index`, returning it.
        pub fn swapRemove(self: *Self, index: usize) T {
            return self.list.swapRemove(index);
        }

        pub fn get(self: *Self, index: usize) *T {
            return &self.list.items[index];
        }

        pub fn len(self: *Self) usize {
            return self.list.items.len;
        }

        pub fn raw(self: *Self) []T {
            return self.list.items;
        }

        pub inline fn isComptime() bool {
            return false;
        }

        pub fn resetAllocator(self: *Self, allocator: *std.mem.Allocator) void {
            self.list.allocator = allocator;
        }

        pub fn clone(self: *Self) !Self {
            // Create a new storage with the same capacity
            var new_self = Self.init(.{self.list.allocator}, self.list.capacity());
            // Copy items
            inline for (self.list.items) |item| {
                try new_self.list.append(item);
            }
            return new_self;
        }
    };
}
