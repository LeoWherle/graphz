const std = @import("std");

pub fn StaticStorage(comptime T: type, comptime Capacity: usize) type {
    return struct {
        pub const Self = @This();
        buffer: [Capacity]T,
        count: usize,

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

        pub fn get(self: *Self, index: usize) *T {
            return &self.buffer[index];
        }

        pub inline fn len(self: *Self) usize {
            return self.count;
        }

        pub inline fn raw(self: *Self) []T {
            return self.buffer[0..self.count];
        }

        pub inline fn isComptime() bool {
            // TODO add check if T is comptime compatible
            return true;
        }
    };
}

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
            @compileError("Allocator is required for DynamicStorage");
        }

        pub inline fn deinit(self: *Self) void {
            self.list.deinit();
        }

        pub inline fn ensureTotalCapacity(self: *Self, capacity: usize) !void {
            try self.list.ensureTotalCapacity(capacity);
        }

        pub inline fn append(self: *Self, value: T) !void {
            try self.list.append(value);
        }

        pub inline fn get(self: *Self, index: usize) *T {
            return &self.list.items[index];
        }

        pub inline fn len(self: *Self) usize {
            return self.list.items.len;
        }

        pub inline fn raw(self: *Self) []T {
            return self.list.items;
        }

        pub inline fn isComptime() bool {
            return false;
        }
    };
}

pub const EdgeType = union(enum) {
    Directed: void,
    Undirected: void,
};

pub const Direction = enum(u8) {
    Outgoing = 0,
    Incoming = 1,

    pub inline fn opposite(self: Direction) Direction {
        return switch (self) {
            .Outgoing => .Incoming,
            .Incoming => .Outgoing,
        };
    }

    pub inline fn index(self: Direction) u32 {
        return (@intFromEnum(self) & 0x1);
    }
};

fn GraphNode(comptime N: type) type {
    return struct {
        const Self = @This();

        value: N,
        next: [2]u32, // [Outgoing, Incoming]

        pub fn nextEdge(self: *const Self, direction: Direction) u32 {
            return self.next[direction.index()];
        }
    };
}

fn GraphEdge(comptime E: type) type {
    return struct {
        const Self = @This();
        const SOURCE_INDEX = 0;
        const TARGET_INDEX = 1;

        value: E,
        next: [2]u32, // [Outgoing, Incoming]
        nodes: [2]u32, // [source, target]

        pub fn nextEdge(self: Self, direction: Direction) u32 {
            return self.next[direction.index()];
        }

        pub fn source(self: *const Self) u32 {
            return self.nodes[SOURCE_INDEX];
        }

        pub fn target(self: *const Self) u32 {
            return self.nodes[TARGET_INDEX];
        }
    };
}

fn GraphNeighbor(comptime E: type, comptime EdgeIndex: type, comptime NodeIndex: type) type {
    comptime {
        // TODO check if EdgeIndex && NodeIndex can be used for indexing
    }
    return struct {
        const Self = @This();

        // The vertex source of the edge
        source: NodeIndex,
        // reference to edges lists std.ArrayList(Edge)
        edges: []GraphEdge(E),
        next_edges: [2]EdgeIndex, // [Outgoing, Incoming]

        pub fn next(self: *Self) ?NodeIndex {
            // Check outgoing edges
            if (self.next_edges[Direction.Outgoing.index()] < self.edges.len) {
                const edge = self.edges[self.next_edges[Direction.Outgoing.index()]];
                self.next_edges[Direction.Outgoing.index()] = edge.next[Direction.Outgoing.index()];
                return edge.target();
            }

            // Check incoming edges
            while (self.next_edges[Direction.Incoming.index()] < self.edges.len) {
                const edge = self.edges[self.next_edges[Direction.Incoming.index()]];
                self.next_edges[Direction.Incoming.index()] = edge.next[Direction.Incoming.index()];
                // For undirected iteration, skip self-loops in the incoming list
                if (edge.source() != self.source) {
                    return edge.source();
                }
            }

            return null;
        }
    };
}

fn Graph(comptime N: type, comptime E: type, comptime Et: EdgeType, comptime NodeStorage: fn (comptime type) type, comptime EdgeStorage: fn (comptime type) type) type {
    return struct {
        const Self = @This();

        const EdgeIndexType = u32;
        const EdgeIndexEnd = std.math.maxInt(EdgeIndexType);
        const NodeIndexType = u32;
        const NodeIndexEnd = std.math.maxInt(NodeIndexType);

        const Node = GraphNode(N);
        const Edge = GraphEdge(E);
        const Neighbor = GraphNeighbor(E, EdgeIndexType, NodeIndexType);
        const Allocator = std.mem.Allocator;

        nodes: NodeStorage(Node),
        edges: EdgeStorage(Edge),
        allocator: ?Allocator,

        pub fn isComptime() bool {
            return NodeStorage(Node).isComptime() and EdgeStorage(Edge).isComptime();
        }

        pub fn init(allocator: ?*Allocator, nodeCapacity: usize, edgeCapacity: usize) Self {
            const needsAllocator = !isComptime();
            return Self{
                .nodes = NodeStorage(Node).init(allocator, nodeCapacity),
                .edges = EdgeStorage(Edge).init(allocator, edgeCapacity),
                .allocator = if (needsAllocator) allocator.?.* else null,
            };
        }

        pub fn deinit(self: *Self) void {
            self.nodes.deinit();
            self.edges.deinit();
        }

        pub fn initCapacity(allocator: ?*Allocator, nodesCapacity: usize, edgesCapacity: usize) Allocator.Error!Self {
            var graph = Self.init(allocator, nodesCapacity, edgesCapacity);
            try graph.nodes.ensureTotalCapacity(nodesCapacity);
            try graph.edges.ensureTotalCapacity(edgesCapacity);
            return graph;
        }

        pub fn resetAllocator(self: Self, allocator: *Allocator) void {
            self.nodes.resetAllocator(allocator);
            self.edges.resetAllocator(allocator);
        }

        pub fn clone(self: Self) Allocator.Error!Self {
            var cloned = try Self.initCapacity(self.allocator, self.nodes.len(), self.edges.len());
            errdefer cloned.deinit();
            cloned.nodes = try self.nodes.clone();
            errdefer cloned.nodes.deinit();
            cloned.edges = try self.edges.clone();
            return cloned;
        }

        pub fn nodes_count(self: Self) usize {
            return self.nodes.len();
        }

        pub fn edges_count(self: Self) usize {
            return self.edges.len();
        }

        pub inline fn isDirected() bool {
            return Et == .Directed;
        }

        pub fn addNode(self: *Self, value: N) !NodeIndexType {
            const node = Node{
                .value = value,
                .next = [2]EdgeIndexType{ EdgeIndexEnd, EdgeIndexEnd },
            };
            const node_idx = @as(NodeIndexType, @intCast(self.nodes.len()));
            try self.nodes.append(node);
            return node_idx;
        }

        pub fn nodeWeight(self: *Self, node_idx: NodeIndexType) ?*N {
            if (node_idx >= self.nodes.len()) return null;
            return &self.nodes.items[node_idx].value;
        }

        pub fn addEdge(self: *Self, source: NodeIndexType, target: NodeIndexType, value: E) !EdgeIndexType {
            const edge_idx = @as(EdgeIndexType, @intCast(self.edges.len()));
            if (source >= self.nodes.len() or target >= self.nodes.len()) {
                return error.NodeIndexOutOfBounds;
            }

            var edge = Edge{
                .value = value,
                .next = [2]EdgeIndexType{ EdgeIndexEnd, EdgeIndexEnd },
                .nodes = [2]NodeIndexType{ source, target },
            };

            // Update node connections
            if (source == target) {
                // Self-loop case
                var node = &self.nodes.get(source).*;
                const old_next = node.next;
                node.next = [2]EdgeIndexType{ edge_idx, edge_idx };
                edge.next = old_next;
            } else {
                // Different nodes case
                var source_node = self.nodes.get(source);
                var target_node = self.nodes.get(target);
                edge.next = [2]EdgeIndexType{ source_node.next[Edge.SOURCE_INDEX], target_node.next[Edge.TARGET_INDEX] };
                source_node.next[Edge.SOURCE_INDEX] = edge_idx;
                target_node.next[Edge.TARGET_INDEX] = edge_idx;
            }

            try self.edges.append(edge);
            return edge_idx;
        }

        pub fn edgeWeight(self: *Self, edge_idx: EdgeIndexType) ?*E {
            if (edge_idx >= self.edges.len()) return null;
            return self.edges.get(edge_idx).value;
        }

        pub fn edgeEndpoints(self: *Self, edge_idx: u32) ?struct { source: NodeIndexType, target: NodeIndexType } {
            if (edge_idx >= self.edges.len()) return null;
            const edge = self.edges.get(edge_idx);
            return .{ .source = edge.source(), .target = edge.target() };
        }

        pub fn updateEdge(self: *Self, source: NodeIndexType, target: NodeIndexType, value: E) !EdgeIndexType {
            if (self.findEdge(source, target)) |edge_idx| {
                if (self.edgeWeight(edge_idx)) |edge_weight| {
                    edge_weight.* = value;
                    return edge_idx;
                }
            }
            return try self.addEdge(source, target, value);
        }

        pub fn removeNode(self: *Self, node_idx: NodeIndexType) ?N {
            if (node_idx >= self.nodes.len()) return null;

            // Remove all edges connected to this node
            inline for ([_]Direction{ .Outgoing, .Incoming }) |dir| {
                const k = dir.index();
                while (true) {
                    const next = self.nodes.get(node_idx).next[k];
                    if (next == EdgeIndexEnd) break;
                    _ = self.removeEdge(next);
                }
            }

            const node = self.nodes.swapRemove(node_idx);

            // If a node was swapped into place, update its edge references
            if (node_idx < self.nodes.len()) {
                const old_index = @as(NodeIndexType, @intCast(self.nodes.len()));
                const new_index = node_idx;

                // Update edge references for the swapped node
                inline for ([_]Direction{ .Outgoing, .Incoming }) |dir| {
                    const k = dir.index();
                    var edge_idx = self.nodes.get(new_index).next[k];
                    while (edge_idx < self.edges.len()) {
                        var edge = self.edges.get(edge_idx);
                        if (edge.nodes[k] == old_index) {
                            edge.nodes[k] = new_index;
                        }
                        edge_idx = edge.next[k];
                    }
                }
            }

            return node.value;
        }

        fn changeEdgeLinks(self: *Self, edge_node: [2]NodeIndexType, e: EdgeIndexType, edge_next: [2]EdgeIndexType) void {
            inline for ([_]Direction{ .Outgoing, .Incoming }) |d| {
                const k = d.index();
                if (edge_node[k] >= self.nodes.len()) {
                    std.debug.assert(false);
                    return;
                }

                var node = &self.nodes.get(edge_node)[k];
                const first = node.next[k];
                if (first == e) {
                    node.next[k] = edge_next[k];
                } else {
                    var edge_idx = first;
                    while (edge_idx < self.edges.len()) {
                        var current_edge = self.edges.get(edge_idx);
                        if (current_edge.next[k] == e) {
                            current_edge.next[k] = edge_next[k];
                            break;
                        }
                        edge_idx = current_edge.next[k];
                    }
                }
            }
        }

        pub fn removeEdge(self: *Self, edge_idx: EdgeIndexType) ?E {
            if (edge_idx >= self.edges.len()) return null;

            // Get the edge's node connections and next pointers
            const edge = self.edges.get(edge_idx);
            const edge_node = edge.nodes;
            const edge_next = edge.next;

            // Remove the edge from its in and out lists
            self.changeEdgeLinks(edge_node, edge_idx, edge_next);

            // Swap remove the edge
            const removed_edge = self.edges.swapRemove(edge_idx);

            // If an edge was swapped into place, update its references
            if (edge_idx < self.edges.len()) {
                const old_index = @as(EdgeIndexType, @intCast(self.edges.len()));
                const new_index = edge_idx;

                const swapped_edge = self.edges.get(new_index);
                self.changeEdgeLinks(swapped_edge.nodes, old_index, [2]EdgeIndexType{ new_index, new_index });
            }

            return removed_edge.value;
        }

        pub fn findEdge(self: *Self, source: NodeIndexType, target: NodeIndexType) ?EdgeIndexType {
            if (comptime !isDirected()) {
                return if (self.findEdgeUndirected(source, target)) |result| result.edge_idx else null;
            } else {
                if (source >= self.nodes.len()) return null;
                const node = self.nodes.get(source);
                return self.findEdgeDirectedFromNode(node, target);
            }
        }

        /// Find an edge from a node to a target node
        pub fn findEdgeDirectedFromNode(self: *Self, node: *Node, target: NodeIndexType) ?EdgeIndexType {
            // Iterate over the outgoing edges of the node
            var edix = node.next[Direction.Outgoing.index()];
            while (edix < self.edges.len()) {
                const edge = self.edges.get(edix);
                // If the edge's target is the target node, we found the edge
                if (edge.nodes[Direction.Incoming.index()] == target) {
                    return edix;
                }
                edix = edge.next[Direction.Outgoing.index()];
            }
            return null;
        }

        const EdgeSearchResult = struct { edge_idx: EdgeIndexType, direction: Direction };

        pub fn findEdgeUndirected(self: *Self, a: NodeIndexType, b: NodeIndexType) ?EdgeSearchResult {
            if (a >= self.nodes.len()) {
                return null;
            }
            const node = self.nodes.get(a);
            return self.findEdgeUndirectedFromNode(node, b);
        }

        pub fn findEdgeUndirectedFromNode(self: *Self, node: *Node, target: NodeIndexType) ?EdgeSearchResult {
            inline for ([_]Direction{ .Outgoing, .Incoming }) |dir| {
                const k = dir.index();
                var edix = node.next[k];

                while (edix < self.edges.len()) {
                    const edge = self.edges.get(edix);
                    if (edge.nodes[1 - k] == target) {
                        return .{ .edge_idx = edix, .direction = dir };
                    }
                    edix = edge.next[k];
                }
            }
            return null;
        }

        pub fn rawNodes(self: *Self) []Node {
            return self.nodes.raw();
        }
        pub fn rawEdges(self: *Self) []Edge {
            return self.edges.raw();
        }

        pub fn neighbors(self: *Self, node_idx: NodeIndexType, comptime dir: ?Direction) Neighbor {
            const empty_next = [2]EdgeIndexType{ EdgeIndexEnd, EdgeIndexEnd };
            var neighbor = Neighbor{
                .source = node_idx,
                .edges = self.edges.raw(),
                .next_edges = empty_next,
            };

            if (node_idx >= self.nodes.len()) {
                return neighbor;
            }

            const node = self.nodes.get(node_idx);

            switch (Et) {
                .Directed => {
                    if (comptime dir) |direction| {
                        neighbor.next_edges[direction.index()] = node.nextEdge(direction);
                    } else {
                        @compileError("Direction must be specified for directed graphs");
                    }
                },
                .Undirected => {
                    neighbor.next_edges = node.next;
                },
            }
            return neighbor;
        }
    };
}

pub fn StaticStorage128(comptime T: type) type {
    return StaticStorage(T, 320);
}

/// Directed graph
pub fn DiGraph(comptime N: type, comptime E: type) type {
    return Graph(N, E, EdgeType{ .Directed = {} }, StaticStorage128, StaticStorage128);
}

/// Undirected graph
pub fn UnGraph(comptime N: type, comptime E: type) type {
    return Graph(N, E, EdgeType{ .Undirected = {} }, StaticStorage128, StaticStorage128);
}

const ResultType = struct {
    graph: UnGraph(u32, u32),
    edge_q: ?u32,
    root_node: u32,
};

pub fn computeGraph() !ResultType {
    var g = try UnGraph(u32, u32).initCapacity(null, 320, 320);
    defer g.deinit();

    const n1 = try g.addNode(1);
    const n2 = try g.addNode(2);
    const n3 = try g.addNode(3);

    _ = try g.addEdge(n1, n2, 12);
    _ = try g.addEdge(n2, n3, 23);
    _ = try g.addEdge(n3, n1, 31);

    const edge_q = g.findEdge(n2, n1);

    return ResultType{ .graph = g, .edge_q = edge_q, .root_node = n1 };
}

pub fn myMain() !void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // var allocator = gpa.allocator();

    const result = comptime try computeGraph();
    var graph = result.graph;
    const edge_q = result.edge_q;
    const n1 = result.root_node;

    if (edge_q) |edge_idx| {
        const edge_data = graph.edges.get(edge_idx);
        std.debug.print("Found edge from {} to {} with value {}\n", .{ edge_data.source(), edge_data.target(), edge_data.value });
    } else {
        std.debug.print("Edge not found\n", .{});
    }

    for (graph.rawEdges()) |edge| {
        std.debug.print("Edge from {} to {} with value {}\n", .{ edge.source(), edge.target(), edge.value });
    }

    var neighbors = graph.neighbors(n1, .Incoming);
    while (neighbors.next()) |neighbor| {
        const data = &graph.nodes.raw()[neighbor];
        std.debug.print("Neighbor: {} with data : {}\n", .{ neighbor, data.value });
    }

    dfs(UnGraph(u32, u32), &graph, n1, null, visitorFunction);
}

pub fn main() !void {
    return myMain();
}

pub fn visitorFunction(context: @TypeOf(null), node: u32) void {
    std.debug.print("Visited node {} {}\n", .{ node, context });
}

pub fn dfs(
    comptime G: type,
    graph: *G,
    start: G.NodeIndex,
    context: anytype,
    comptime visitor: fn (@TypeOf(context), G.NodeIndex) void,
) void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // if graph has allocator, use it else use std.mem.heap
    const allocator = if (!G.isComptime()) graph.allocator orelse gpa.allocator() else gpa.allocator();

    var visited = std.AutoHashMap(G.NodeIndex, void).init(allocator);
    defer visited.deinit();

    const NeighborIterator = struct {
        neighbors: G.Neighbor,
    };

    var stack = std.ArrayList(NeighborIterator).init(allocator);
    defer stack.deinit();

    // Mark start as discovered
    visited.put(start, {}) catch return;
    visitor(context, start);

    // Push iterator of adjacent edges
    stack.append(.{
        .neighbors = graph.neighbors(start, if (comptime G.isDirected()) .Outgoing else null),
    }) catch return;

    while (stack.items.len > 0) {
        const current_iter = &stack.items[stack.items.len - 1];

        if (current_iter.neighbors.next()) |neighbor| {
            if (!visited.contains(neighbor)) {
                // Mark as discovered and visit
                visited.put(neighbor, {}) catch return;
                visitor(context, neighbor);

                // Push iterator for the new node
                stack.append(.{
                    .neighbors = graph.neighbors(neighbor, if (G.isDirected()) .Outgoing else null),
                }) catch return;
            }
        } else {
            _ = stack.pop();
        }
    }
}
