import sys

from adjacency_list import Graph


def prims(graph, root_index):
    """
    Given a graph and a zero-indexed root integer, returns Prim's minimum spanning tree .

    Prim's is a greedy algroithm that constructs a Minimum Spanning Tree over a
    weighted graph. A MST is an acyclic traversal of the graph that includes
    every node in the graph. There is no other tree which spans the graph and has
    a small sum of edge weights.

    Skiena writes, 'the natural greedy algorithm for minimum spanning tree
    repeatedly selects the smallest weight edge that will enlarge the tree.'

    Psudocode:

    algorithm Prims-MST(graph, root):
        prims_tree = root
        while there are nodes in the graph that are not in prims_tree:
            select the edge with the smallest weight that connects a new node to
            prims_tree.
            Add that node to prims_tree
        return prims_tree

    Implementation:

    Skiena presents a two-stage implementation of Prim's. For every vertext:
        We look at the consequences of adding that vertex's adjacent nodes to
        the tree. For Prim's, this means updating a list of edge costs.

        The second stage is:
        Then, for each vertex, we look at the associated cost of adding it into
        the tree, and pick the cheapest vertex to add.

    Run-time:
    We have a loop over all nodes, and nested inside that is a loop over all
    nodes. O(n^2)

    There are better implementations that use priority queues to speed up Prim's.
    """
    # Initialization
    # tree = {index: None for index in range(graph.nnodes)}
    nodes_in_tree = [False for _ in range(graph.nnodes)]
    distance_from_tree = [sys.maxsize for _ in range(graph.nnodes)]
    parents = [None for _ in range(graph.nnodes)]
    distance_from_tree[root_index] = 0

    while not nodes_in_tree[root_index]:
        # First stage, what are the consequences of adding the new root vertex
        # to the tree?
        nodes_in_tree[root_index] = True
        vertex = graph.nodes[root_index]
        for adjacent_edge in vertex:
            weight = adjacent_edge.weight
            if weight < distance_from_tree[adjacent_edge.y] and not nodes_in_tree[adjacent_edge.y]:
                distance_from_tree[adjacent_edge.y] = weight
                parents[adjacent_edge.y] = root_index

        # Second stage, find the cheapest vertex to add to the tree:
        root_index = 0
        distance = sys.maxsize
        for node_index, node_in_tree in enumerate(nodes_in_tree):
            if not node_in_tree and distance_from_tree[node_index] < distance:
                distance = distance_from_tree[node_index]
                root_index = node_index
    return parents


test_graph = Graph(directed=False)
test_graph.add_edge(0, 1, weight=12)
test_graph.add_edge(0, 2, weight=7)
test_graph.add_edge(0, 3, weight=5)
test_graph.add_edge(1, 2, weight=4)
test_graph.add_edge(1, 5, weight=7)
test_graph.add_edge(2, 3, weight=9)
test_graph.add_edge(2, 4, weight=4)
test_graph.add_edge(2, 5, weight=3)
test_graph.add_edge(3, 4, weight=7)
test_graph.add_edge(4, 5, weight=2)
test_graph.add_edge(4, 6, weight=5)
test_graph.add_edge(5, 6, weight=2)

# parents = prims(test_graph, 0)


def trace_mst_parents(parents):
    """ Given a parents list of a graph, converts that to the (x,y) indexed edges in the graph. """
    edges = set()

    def inner(index, parent):
        while parent is not None:
            edges.add((index, parent))
            index, parent = parents[parent]

    parents = [(index, item) for index, item in enumerate(parents)]
    for index, parent in parents:
        inner(index, parent)
    return edges


def tree_weight(graph, root):
    parents = prims(graph, root)
    edges = [graph.get_edge(*edge) for edge in trace_mst_parents(parents)]
    return sum(edge.weight for edge in edges)


for root in range(test_graph.nnodes):
    assert tree_weight(test_graph, root) == 23

print('done')
