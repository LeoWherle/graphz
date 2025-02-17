const std = @import("std");

pub fn VisitorFn(comptime ContextType: type, comptime NodeIndexType: type) type {
    return fn (context: ContextType, node: NodeIndexType) void;
}

// TODO: make it work at comptime
pub fn dfs(
    comptime G: type,
    graph: *G,
    start: G.NodeIndexType,
    context: anytype,
    comptime visitor: VisitorFn(@TypeOf(context), G.NodeIndexType),
) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const stack_alloc = if (!G.isComptime()) (if (graph.allocator) |a| a.* else gpa.allocator()) else gpa.allocator();

    var visited = std.AutoHashMap(G.NodeIndexType, bool).init(stack_alloc);
    defer visited.deinit();

    const NeighborIter = struct {
        neighbors: G.Neighbor,
    };
    var stack = std.ArrayList(NeighborIter).init(stack_alloc);
    defer stack.deinit();

    // Mark start as visited
    try visited.put(start, true);
    visitor(context, start);

    // For directed graphs, only walk outgoing edges
    const search_dir = if (G.isDirected()) .Outgoing else null;

    try stack.append(.{ .neighbors = graph.neighbors(start, search_dir) });

    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.neighbors.next()) |nbr| {
            if (!visited.contains(nbr)) {
                // Mark visited
                try visited.put(nbr, true);
                visitor(context, nbr);

                try stack.append(.{ .neighbors = graph.neighbors(nbr, search_dir) });
            }
        } else {
            _ = stack.pop();
        }
    }
}

// TODO: make it work at comptime
pub fn bfs(
    comptime G: type,
    graph: *G,
    start: G.NodeIndexType,
    context: anytype,
    comptime visitor: VisitorFn(@TypeOf(context), G.NodeIndexType),
) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const queue_alloc = if (!G.isComptime()) (if (graph.allocator) |a| a.* else gpa.allocator()) else gpa.allocator();

    var visited = std.AutoHashMap(G.NodeIndexType, bool).init(queue_alloc);
    defer visited.deinit();

    var queue = std.ArrayList(G.NodeIndexType).init(queue_alloc);
    defer queue.deinit();

    // Mark start as visited
    try visited.put(start, true);
    visitor(context, start);
    try queue.append(start);

    // For directed graphs, only follow outgoing edges
    const search_dir = if (G.isDirected()) .Outgoing else null;

    while (queue.items.len > 0) {
        const current = queue.items[0];
        _ = queue.orderedRemove(0);

        var neighbors_iter = graph.neighbors(current, search_dir);
        while (neighbors_iter.next()) |nbr| {
            if (!visited.contains(nbr)) {
                try visited.put(nbr, true);
                visitor(context, nbr);
                try queue.append(nbr);
            }
        }
    }
}
