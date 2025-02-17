const std = @import("std");
const storage = @import("storage.zig");
const node = @import("node.zig");

/// The main Graph type, parameterized by:
///  - `N`: Node payload type
///  - `E`: Edge payload type
///  - `Et`: EdgeType (Directed or Undirected)
///  - `NodeStorage`: how to store nodes (static or dynamic)
///  - `EdgeStorage`: how to store edges (static or dynamic)
pub fn Graph(
    comptime N: type,
    comptime E: type,
    comptime Et: node.EdgeType,
    comptime NodeStorage: fn (comptime type) type,
    comptime EdgeStorage: fn (comptime type) type,
) type {
    return struct {
        const Self = @This();

        // Indices
        pub const EdgeIndexType = u32;
        const EdgeIndexEnd = std.math.maxInt(EdgeIndexType);
        pub const NodeIndexType = u32;
        const NodeIndexEnd = std.math.maxInt(NodeIndexType);

        // Node, Edge, and neighbor definitions
        pub const Node = node.GraphNode(N);
        pub const Edge = node.GraphEdge(E);
        pub const Neighbor = node.GraphNeighbor(E, EdgeIndexType, NodeIndexType);

        pub const Allocator = std.mem.Allocator;

        // Actual storage
        nodes: NodeStorage(Node),
        edges: EdgeStorage(Edge),
        allocator: ?*Allocator,

        /// Distinguish if this Graph can be used at comptime
        pub fn isComptime() bool {
            return NodeStorage(Node).isComptime() and EdgeStorage(Edge).isComptime();
        }

        /// Initialize the graph, but doesn't ensure capacity.
        /// For run-time usage, pass a valid allocator if `isComptime() == false`.
        pub fn init(allocator: ?*Allocator, nodeCapacity: usize, edgeCapacity: usize) Self {
            const needs_allocator = !isComptime();
            return Self{
                .nodes = NodeStorage(Node).init(allocator, nodeCapacity),
                .edges = EdgeStorage(Edge).init(allocator, edgeCapacity),
                .allocator = if (needs_allocator) allocator else null,
            };
        }

        /// Deallocate resources if needed
        pub fn deinit(self: *Self) void {
            self.nodes.deinit();
            self.edges.deinit();
        }

        /// Initialize + ensure capacity (throws if OOM).
        pub fn initCapacity(allocator: ?*Allocator, nodesCap: usize, edgesCap: usize) Allocator.Error!Self {
            var g = Self.init(allocator, nodesCap, edgesCap);
            try g.nodes.ensureTotalCapacity(nodesCap);
            try g.edges.ensureTotalCapacity(edgesCap);
            return g;
        }

        pub fn resetAllocator(self: *Self, allocator: *Allocator) void {
            self.nodes.resetAllocator(allocator);
            self.edges.resetAllocator(allocator);
            self.allocator = allocator;
        }

        /// Clone the entire graph
        pub fn clone(self: *Self) Allocator.Error!Self {
            var cloned = try Self.initCapacity(self.allocator, self.nodes.len(), self.edges.len());
            errdefer cloned.deinit();

            cloned.nodes = try self.nodes.clone();
            errdefer cloned.nodes.deinit();

            cloned.edges = try self.edges.clone();
            // no defer needed if successful

            return cloned;
        }

        pub fn nodesCount(self: *Self) usize {
            return self.nodes.len();
        }
        pub fn edgesCount(self: *Self) usize {
            return self.edges.len();
        }

        pub inline fn isDirected() bool {
            return Et == .Directed;
        }

        /// Add a node to the graph, returning its numeric index.
        pub fn addNode(self: *Self, value: N) !NodeIndexType {
            const node_idx = @as(NodeIndexType, self.nodes.len());
            // Create a node with no edges
            const gnode = Node{ .value = value, .next = [2]EdgeIndexType{ EdgeIndexEnd, EdgeIndexEnd } };
            try self.nodes.append(gnode);
            return node_idx;
        }

        /// Return a pointer to the node payload, or null if out of range.
        pub fn nodeWeight(self: *Self, idx: NodeIndexType) ?*N {
            if (idx >= self.nodes.len()) return null;
            return &self.nodes.raw()[idx].value;
        }

        /// Add an edge to the graph between `source` and `target` with payload `value`.
        pub fn addEdge(self: *Self, source: NodeIndexType, target: NodeIndexType, value: E) !EdgeIndexType {
            if (source >= self.nodes.len() or target >= self.nodes.len()) {
                return error.NodeIndexOutOfBounds;
            }
            const edge_idx = @as(EdgeIndexType, self.edges.len());

            var edge = Edge{
                .value = value,
                .next = [2]EdgeIndexType{ EdgeIndexEnd, EdgeIndexEnd },
                .nodes = [2]NodeIndexType{ source, target },
            };

            if (source == target) {
                // Self-loop
                var node_ref = &self.nodes.raw()[source];
                const old_next = node_ref.next;
                node_ref.next = [2]EdgeIndexType{ edge_idx, edge_idx };
                edge.next = old_next;
            } else {
                // Normal edge
                var source_node = &self.nodes.raw()[source];
                var target_node = &self.nodes.raw()[target];
                edge.next = [2]EdgeIndexType{ source_node.next[0], target_node.next[1] };
                source_node.next[0] = edge_idx;
                target_node.next[1] = edge_idx;
            }

            try self.edges.append(edge);
            return edge_idx;
        }

        /// Retrieve a pointer to edge payload by index, or null if out of range.
        pub fn edgeWeight(self: *Self, idx: EdgeIndexType) ?*E {
            if (idx >= self.edges.len()) return null;
            return &self.edges.raw()[idx].value;
        }

        /// Return the (source, target) of the edge, if valid.
        pub fn edgeEndpoints(self: *Self, idx: EdgeIndexType) ?struct { source: NodeIndexType, target: NodeIndexType } {
            if (idx >= self.edges.len()) return null;
            const e = self.edges.raw()[idx];
            return .{ .source = e.source(), .target = e.target() };
        }

        /// Update or create an edge, returning its index.
        pub fn updateEdge(self: *Self, source: NodeIndexType, target: NodeIndexType, value: E) !EdgeIndexType {
            if (self.findEdge(source, target)) |edge_idx| {
                if (self.edgeWeight(edge_idx)) |w| {
                    w.* = value;
                    return edge_idx;
                }
            }
            return try self.addEdge(source, target, value);
        }

        /// Remove a node, returning its old payload, or null if out of range.
        pub fn removeNode(self: *Self, idx: NodeIndexType) ?N {
            if (idx >= self.nodes.len()) return null;

            // Remove all edges connected to this node
            inline for ([_]node.Direction{ .Outgoing, .Incoming }) |dir| {
                const k = dir.index();
                while (true) {
                    const next_ed = self.nodes.raw()[idx].next[k];
                    if (next_ed == EdgeIndexEnd) break;
                    _ = self.removeEdge(next_ed);
                }
            }

            // Swap-remove node
            const removed_node = self.nodes.swapRemove(idx);

            // If we swapped a node into position `idx`, fix up its edges
            if (idx < self.nodes.len()) {
                const old_index = @as(NodeIndexType, self.nodes.len());
                const new_index = idx;

                // Update edges that pointed to old_index
                inline for ([_]node.Direction{ .Outgoing, .Incoming }) |dir| {
                    const k = dir.index();
                    var ed = self.nodes.raw()[new_index].next[k];
                    while (ed < self.edges.len()) {
                        var e_ref = &self.edges.raw()[ed];
                        if (e_ref.nodes[k] == old_index) {
                            e_ref.nodes[k] = new_index;
                        }
                        ed = e_ref.next[k];
                    }
                }
            }
            return removed_node.value;
        }

        /// Remove an edge, returning its old payload, or null if out of range.
        pub fn removeEdge(self: *Self, idx: EdgeIndexType) ?E {
            if (idx >= self.edges.len()) return null;
            const e_ref = &self.edges.raw()[idx];
            const edge_node = e_ref.nodes;
            const edge_next = e_ref.next;

            // Unlink this edge from the two node adjacency lists
            self.changeEdgeLinks(edge_node, idx, edge_next);

            // Perform swap-remove
            const removed_edge = self.edges.swapRemove(idx);

            // If we swapped an edge into position `idx`, we must fix it.
            if (idx < self.edges.len()) {
                const old_idx = @as(EdgeIndexType, self.edges.len());
                const new_idx = idx;
                const swapped_edge = &self.edges.raw()[new_idx];
                self.changeEdgeLinks(swapped_edge.nodes, old_idx, .{ new_idx, new_idx });
            }

            return removed_edge.value;
        }

        fn changeEdgeLinks(self: *Self, edge_node: [2]NodeIndexType, e: EdgeIndexType, edge_next: [2]EdgeIndexType) void {
            inline for ([_]node.Direction{ .Outgoing, .Incoming }) |dir| {
                const k = dir.index();
                std.debug.assert(edge_node[k] < self.nodes.len());
                var node_ref = &self.nodes.raw()[edge_node[k]];
                const first_ed = node_ref.next[k];

                if (first_ed == e) {
                    node_ref.next[k] = edge_next[k];
                } else {
                    var walk = first_ed;
                    while (walk < self.edges.len()) {
                        var curr = &self.edges.raw()[walk];
                        if (curr.next[k] == e) {
                            curr.next[k] = edge_next[k];
                            break;
                        }
                        walk = curr.next[k];
                    }
                }
            }
        }

        /// Find an edge by (source, target) or (a, b) in undirected.
        pub fn findEdge(self: *Self, a: NodeIndexType, b: NodeIndexType) ?EdgeIndexType {
            if (comptime !isDirected()) {
                return if (self.findEdgeUndirected(a, b)) |res| res.edge_idx else null;
            } else {
                if (a >= self.nodes.len()) return null;
                return self.findEdgeDirectedFromNode(&self.nodes.raw()[a], b);
            }
        }

        fn findEdgeDirectedFromNode(self: *Self, node_ptr: *Node, target: NodeIndexType) ?EdgeIndexType {
            var e_idx = node_ptr.next[0]; // outgoing
            while (e_idx < self.edges.len()) {
                const e = &self.edges.raw()[e_idx];
                if (e.nodes[1] == target) {
                    return e_idx;
                }
                e_idx = e.next[0];
            }
            return null;
        }

        const EdgeSearchResult = struct {
            edge_idx: EdgeIndexType,
            direction: node.Direction,
        };

        fn findEdgeUndirected(self: *Self, a: NodeIndexType, b: NodeIndexType) ?EdgeSearchResult {
            if (a >= self.nodes.len()) return null;
            return self.findEdgeUndirectedFromNode(&self.nodes.raw()[a], b);
        }

        fn findEdgeUndirectedFromNode(self: *Self, node_ptr: *Node, target: NodeIndexType) ?EdgeSearchResult {
            // Check outgoing, then incoming
            inline for ([_]node.Direction{ .Outgoing, .Incoming }) |dir| {
                var e_idx = node_ptr.next[dir.index()];
                while (e_idx < self.edges.len()) {
                    const e = &self.edges.raw()[e_idx];
                    if (e.nodes[1 - dir.index()] == target) {
                        return .{ .edge_idx = e_idx, .direction = dir };
                    }
                    e_idx = e.next[dir.index()];
                }
            }
            return null;
        }

        /// Raw access to node array
        pub fn rawNodes(self: *Self) []Node {
            return self.nodes.raw();
        }

        /// Raw access to edge array
        pub fn rawEdges(self: *Self) []Edge {
            return self.edges.raw();
        }

        /// Acquire a neighbor-iterator for a given node index.
        ///
        /// - For an undirected graph, `dir` is ignored and we iterate both ways.
        /// - For a directed graph, you must specify `.Outgoing` or `.Incoming`.
        pub fn neighbors(self: *Self, node_idx: NodeIndexType, comptime dir: ?node.Direction) Neighbor {
            var neighbor = Neighbor{
                .source = node_idx,
                .edges = self.edges.raw(),
                .next_edges = [2]EdgeIndexType{ EdgeIndexEnd, EdgeIndexEnd },
            };

            if (node_idx >= self.nodes.len()) {
                return neighbor;
            }

            const node_ref = &self.nodes.raw()[node_idx];

            switch (Et) {
                .Directed => {
                    if (comptime dir) |direction| {
                        neighbor.next_edges[direction.index()] = node_ref.next[direction.index()];
                    } else {
                        @compileError("Direction must be specified for directed graphs");
                    }
                },
                .Undirected => {
                    neighbor.next_edges = node_ref.next;
                },
            }
            return neighbor;
        }
    };
}

/// A specialized static storage for up to 320 items
pub fn StaticStorage128(comptime T: type) type {
    return storage.StaticStorage(T, 320);
}

/// Convenience type: Directed graph with static storage
pub fn DiGraph(comptime N: type, comptime E: type) type {
    return Graph(N, E, node.EdgeType{ .Directed = {} }, StaticStorage128, StaticStorage128);
}

/// Convenience type: Undirected graph with static storage
pub fn UnGraph(comptime N: type, comptime E: type) type {
    return Graph(N, E, node.EdgeType{ .Undirected = {} }, StaticStorage128, StaticStorage128);
}
