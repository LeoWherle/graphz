const std = @import("std");
const graph = @import("graph.zig");
const algorithms = @import("algorithms.zig");

const ResultType = struct {
    g: graph.UnGraph(u32, u32),
    edge_q: ?u32,
    root_node: u32,
    second_node: u32,
};

const built = blk: {
    // Build a small graph at compile-time
    var g = graph.UnGraph(u32, u32).init(null, 320, 320);
    const n1 = g.addNode(1) catch unreachable;
    const n2 = g.addNode(2) catch unreachable;
    const n3 = g.addNode(3) catch unreachable;

    _ = g.addEdge(n1, n2, 12) catch unreachable;
    _ = g.addEdge(n2, n3, 23) catch unreachable;
    _ = g.addEdge(n3, n1, 31) catch unreachable;

    const edge_q = g.findEdge(n2, n1);

    break :blk ResultType{ .g = g, .edge_q = edge_q, .root_node = n1, .second_node = n2 };
};

pub fn main() !void {
    var g = built.g;
    const edge_q = built.edge_q;
    const n1 = built.root_node;
    const n2 = built.second_node;

    if (edge_q) |edge_idx| {
        const e = g.rawEdges()[edge_idx];
        std.debug.print("Found edge from {} to {} with value = {}\n", .{ e.source(), e.target(), e.value });
    } else {
        std.debug.print("Edge not found.\n", .{});
    }

    // List out edges
    for (g.rawEdges()) |edge| {
        std.debug.print("Edge from {} to {} with value {}\n", .{ edge.source(), edge.target(), edge.value });
    }

    // List incoming neighbors of n1
    std.debug.print("\nIncoming neighbors of node {}:\n", .{n1});
    var neigh_iter = g.neighbors(n1, .Incoming);
    while (neigh_iter.next()) |nbr| {
        const node_data = g.rawNodes()[nbr];
        std.debug.print("  neighbor {} with data = {}\n", .{ nbr, node_data.value });
    }

    // Perform a DFS from n1
    std.debug.print("\nDFS from node {}:\n", .{n1});
    try algorithms.dfs(
        @TypeOf(g),
        &g,
        n1,
        null,
        struct {
            fn visit(_: @TypeOf(null), node_idx: u32) void {
                std.debug.print("  visited node {}\n", .{node_idx});
            }
        }.visit,
    );

    // Perform a BFS from n2
    std.debug.print("\nBFS from node {}:\n", .{n2});
    try algorithms.bfs(
        @TypeOf(g),
        &g,
        n2,
        null,
        struct {
            fn visit(_: @TypeOf(null), node_idx: u32) void {
                std.debug.print("  visited node {}\n", .{node_idx});
            }
        }.visit,
    );
}
