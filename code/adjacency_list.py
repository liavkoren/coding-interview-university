import attr
from attr.validators import instance_of


@attr.s
class EdgeNode:
    """
    An Edge & a Node.

    The source node is the index of the head in the Graph's edges list.
    """
    y = attr.ib(instance_of(int))
    weight = attr.ib(default=1)
    # edges = attr.ib(default=attr.Factory(list), repr=False)
    next = attr.ib(default=None)

    def degree(self):
        out = 0
        while self.next:
            out += 1
            self = self.next
        return out
    def __str__(self):
        return f'EdgeNode(y={self.y}, w={self.weight})'

@attr.s
class Graph:
    edges = attr.ib(default=attr.Factory(list))
    directed = attr.ib(instance_of(bool))
    nedges = attr.ib(0)

    def __init__(self):
        import pdb; pdb.set_trace()
        self.nedges = 0
        for node in self.edges:
            self.nedges += node.degree

    def __str__(self):
        out = []
        for index, edgenode in enumerate(self.edges):
            out.append(f'{index}:\n')
            while edgenode:
                out.append((f'  {edgenode}\n'))
                edgenode = edgenode.next
        return ''.join(out)

    @property
    def nvertices(self):
        return len(self.edges)

    def add_edge(self, x, y, weight=1, directed=False):
        while max(x, y) + 1 > len(self.edges):
            self.edges.append(None)
        edgenode = EdgeNode(y=y, weight=weight, next=self.edges[x])
        self.edges[x] = edgenode
        if directed:
            self.nedges += 1
        else:
            self.add_edge(y, x, weight, directed=True)


graph = Graph(directed=False)
graph.add_edge(0, 1)
graph.add_edge(0, 4)
graph.add_edge(1, 2)
graph.add_edge(1, 3)
graph.add_edge(1, 4)
graph.add_edge(2, 3)
graph.add_edge(3, 4)

print(graph)
