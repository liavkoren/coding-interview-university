from collections import deque
from enum import Enum

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

    def __str__(self):
        return f'EdgeNode(y={self.y}, w={self.weight})'

    def __iter__(self):
        yield self
        while self.next:
            yield self.next
            self = self.next

    def degree(self):
        out = 0
        while self.next:
            out += 1
            self = self.next
        return out


@attr.s
class Graph:
    nodes = attr.ib(default=attr.Factory(list))
    directed = attr.ib(instance_of(bool))
    nedges = attr.ib(0)

    def __str__(self):
        out = []
        for index, edgenode in enumerate(self.nodes):
            out.append(f'{index}:\n')
            while edgenode:
                out.append(f'  {edgenode}\n')
                edgenode = edgenode.next
        return ''.join(out)

    @property
    def nnodes(self):
        return len(self.nodes)

    def add_edge(self, x, y, weight=1, directed=False):
        while max(x, y) + 1 > len(self.nodes):
            self.nodes.append(None)
        edgenode = EdgeNode(y=y, weight=weight, next=self.nodes[x])
        self.nodes[x] = edgenode
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
assert graph.nnodes == 5
assert graph.nedges == 7

graph2 = Graph(directed=False)
graph2.add_edge(0, 5)
graph2.add_edge(0, 1)
graph2.add_edge(1, 4)
graph2.add_edge(1, 2)
graph2.add_edge(2, 3)
graph2.add_edge(3, 4)
graph2.add_edge(4, 0)
assert graph2.nnodes == 6
assert graph2.nedges == 7


def BFS(graph, start):
    """
    Breadth-first search on a graph.

    Args:
        graph (Graph)
        start (int): The node to start at

    Returns:
        parents (list): A list of the index of each node's parent node.
    """
    class States(Enum):
        undiscovered = 0
        discovered = 1
        processed = 2

    node_states = [States.undiscovered] * graph.nnodes
    node_parents = [None] * graph.nnodes
    queue = deque([start])
    node_states[start] = States.discovered

    while queue:
        v = queue.popleft()
        print(f'Processing node #{v:02}')
        node_states[v] = States.processed
        adjacent_nodes = graph.nodes[v]
        for node in adjacent_nodes:
            if node_states[node.y] != States.processed or graph.directed:
                print(f'    Processing edge ({v:02}, {node.y:02})')

            if node_states[node.y] is States.undiscovered:
                queue.append(node.y)
                node_states[node.y] = States.discovered
                node_parents[node.y] = v
        print(f'Leaving node #{v:02}')
    return node_parents


BFS(graph2, 0)

print('----------------------')


def DFS(graph, start):
    class States(Enum):
        undiscovered = 0
        discovered = 1
        processed = 2

    time = 0
    node_states = [States.undiscovered] * graph.nnodes
    node_parents = [None] * graph.nnodes
    entry_times = [0] * graph.nnodes
    exit_times = [0] * graph.nnodes

    def recurse(v):
        nonlocal time
        time += 1
        node_states[v] = States.discovered
        entry_times[v] = time
        print(f'Processing node #{v:02}')
        adjacent_nodes = graph.nodes[v]
        for node in adjacent_nodes:
            if node_states[node.y] is States.undiscovered:
                node_parents[node.y] = v
                print(f'    Processing edge ({v:02}, {node.y:02})')
                recurse(node.y)

            elif node_states[node.y] is States.discovered or graph.directed:
                print(f'   Processing edge ({v:02}, {node.y:02})')
        print(f'Leaving node #{v:02}')
        time += 1
        exit_times[v] = time
        node_states[v] = States.processed

    recurse(start)
    print(f'Parents: {node_parents}\nEntry times: {entry_times}\nExit times{exit_times}')


DFS(graph2, 0)


'''
Other important algos:
- count components
- two-coloring/n-coloring
'''
