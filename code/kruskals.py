import heapq

import attr
from attr.validators import instance_of


@attr.s
class SetUnion:
    """ SetUnion data structure, based on Skiena's implementation, pp 199-200, TADM. """
    parents = attr.ib(instance_of(list))
    sizes = attr.ib(instance_of(list))
    n = attr.ib(instance_of(int))

    def __attrs_post_init__(self):
        self.parents = list(range(self.n))
        self.sizes = [1] * self.n

    def get_root(self, x):
        """ Given an int, returns the root of the set partition that int belongs to. """
        if self.parents[x] == x:
            return x
        return self.get_root(self.parents[x])

    def merge(self, x, y):
        r1, r2 = self.get_root(x), self.get_root(y)

        if r1 == r2:
            return
        if self.sizes[r1] >= self.sizes[r2]:
            self.sizes[r1] = self.sizes[r1] + self.sizes[r2]
            self.parents[r2] = r1
        else:
            self.sizes[r2] = self.sizes[r1] + self.sizes[r2]
            self.parents[r1] = r2

    def same_component(self, x, y):
        return self.get_root(x) == self.get_root(y)


def kruskals(graph):
    """
    Give a graph, returns the parent's list for Kruskal's MST.

    Kruskal's is a really lovely, minimal greedy algo that uses the SetUnion data
    structure to build a MST. It's both faster than Prim's on sparse graphs and
    has a certain elegance and flair that Prim's lacks.

    Psudocode:

    algorithm Kruskals-MST(graph):
        sort the edges by weight using a fast heap O(m)  # this is important, and a step I forgot.
        for every edge in the graph: O(m)
            If the edge connects two components that are disjoint, add the edge
            to the tree. O(ln m)
        return the tree

    Skeina gives the following slightly more concrete psudocode:
    order the edges with a priority queue.
    count = 0
    while count < n - 1:
        get the next edge
        if component(v) != component(w):
            add the edge to the tree
            merge component(v) and component(w) using SetUnion data structure
            count++

    It's the merging components stage that adds the O(ln m) complexity and that
    gives Kruskal's a total complexity of O(m ln m)
    """

    edges = list((edge.weight, edge) for edge in graph.edges)
    heapq.heapify(edges)
    kruskals_mst = list()
    setunion = SetUnion(parents=[], sizes=[], n=graph.nnodes)
    count = 0
    total = 0

    while count < graph.nnodes - 1:
        _, edge = heapq.heappop(edges)
        if not setunion.same_component(edge.x, edge.y):
            kruskals_mst.append(edge)
            setunion.merge(edge.x, edge.y)
            count += 1
            total += edge.weight

    print(f"kruskal's total: {total}")
    return kruskals_mst
