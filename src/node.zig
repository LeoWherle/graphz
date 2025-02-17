const std = @import("std");

/// Whether edges are directed or undirected
pub const EdgeType = union(enum) {
    Directed: void,
    Undirected: void,
};

/// Used for indexing outgoing/incoming edges in adjacency lists
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

/// A graph node, storing `value` plus adjacency pointers for [outgoing, incoming]
pub fn GraphNode(comptime N: type) type {
    return struct {
        value: N,
        // next[0] = index of first outgoing edge
        // next[1] = index of first incoming edge
        next: [2]u32,

        pub fn nextEdge(self: *const @This(), direction: Direction) u32 {
            return self.next[direction.index()];
        }
    };
}

/// A graph edge, storing `value`, next pointers, and its connected [source, target] nodes.
pub fn GraphEdge(comptime E: type) type {
    return struct {
        const SOURCE_INDEX = 0;
        const TARGET_INDEX = 1;

        value: E,
        // next[0] = next outgoing edge, next[1] = next incoming edge
        next: [2]u32,
        // nodes[0] = source, nodes[1] = target
        nodes: [2]u32,

        pub fn nextEdge(self: *const @This(), direction: Direction) u32 {
            return self.next[direction.index()];
        }

        pub fn source(self: *const @This()) u32 {
            return self.nodes[SOURCE_INDEX];
        }

        pub fn target(self: *const @This()) u32 {
            return self.nodes[TARGET_INDEX];
        }
    };
}

/// A helper for iterating neighbors of a node in an undirected or directed sense
pub fn GraphNeighbor(comptime E: type, comptime EdgeIndex: type, comptime NodeIndex: type) type {
    return struct {
        const Self = @This();

        // The node for which we are listing neighbors
        source: NodeIndex,
        // The entire edges array (so we can walk the linked list)
        edges: []GraphEdge(E),
        // next_edges[0] = next outgoing, next_edges[1] = next incoming
        next_edges: [2]EdgeIndex,

        /// Returns the next neighbor node index, or null if there are no more.
        pub fn next(self: *Self) ?NodeIndex {
            // Outgoing edges first
            if (self.next_edges[Direction.Outgoing.index()] < self.edges.len) {
                const edge = self.edges[self.next_edges[Direction.Outgoing.index()]];
                self.next_edges[Direction.Outgoing.index()] = edge.next[Direction.Outgoing.index()];
                return edge.target();
            }

            // Then incoming edges (for undirected). For directed, we might not even call this.
            while (self.next_edges[Direction.Incoming.index()] < self.edges.len) {
                const edge = self.edges[self.next_edges[Direction.Incoming.index()]];
                self.next_edges[Direction.Incoming.index()] = edge.next[Direction.Incoming.index()];
                // For undirected, skip self-loops in incoming
                if (edge.source() != self.source) {
                    return edge.source();
                }
            }

            return null;
        }
    };
}
