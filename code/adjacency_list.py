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


def process_edge(x, y, parent_list):
    print(f'    Processing edge ({v:02}, {node.y:02})')
    # Why do I need to do this check aginst (x,y) / (y, x) when Skiena doesn't include it?
    if parent_list[x] == y or parent_list[y] == x:
        return
    print(f'    Found cycle from {x} to {y}: ')
    find_path(x, y, parent_list)


def find_path(src, dst, parent_list):
    if src == dst or dst is None:
        pass
    else:
        # Do not ignore the order of recursive calls w/r/t other code! Code after
        # a recursive call is deferred.
        find_path(parent_list[src], dst, parent_list)
        print(f'\t{parent_list[src]}')


def DFS(graph, start, process_node_early, process_node_late, process_edge):
    class States(Enum):
        undiscovered = 0
        discovered = 1
        processed = 2

    def edge_classification(x, y):
        pass

    if not hasattr(graph, 'reachable_ancestors'):
        graph.reachable_ancestors = [None] * graph.nnodes
    if not hasattr(graph, 'out_degrees'):
        graph.out_degrees = [None] * graph.nnodes

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
        process_node_early(v)
        adjacent_nodes = graph.nodes[v]
        for node in adjacent_nodes:
            if node_states[node.y] is States.undiscovered:
                node_parents[node.y] = v
                process_edge(v, node.y, node_parents)
                recurse(node.y)
            elif node_states[node.y] is States.discovered or graph.directed:
                process_edge(v, node.y, node_parents)
        process_node_late(v)
        time += 1
        exit_times[v] = time
        node_states[v] = States.processed

    recurse(start)
    print(f'Parents: {node_parents}\nEntry times: {entry_times}\nExit times{exit_times}')


def process_node_early(index):
    print(f'Entering node #{v:02}')


def process_node_late(index):
    print(f'Leaving node #{v:02}')
    graph.reachable_ancestors[index] = index


DFS(graph2, 0, process_node_early, process_node_late, process_edge)

# ----------------------------
# A more literal translation of Skiena's DFS:


discovered = [False] * graph2.nnodes
processed = [False] * graph2.nnodes
entry_times = [0] * graph2.nnodes
exit_times = [0] * graph2.nnodes
parents = [None] * graph2.nnodes
time = 0


def DFS2(graph, v):
    global discovered
    global processed
    global entry_times
    global exit_times
    global parents
    global time

    p = graph.nodes[v]
    discovered[v] = True
    time += 1
    entry_times[v] = time
    print(f'Entering {v}')

    while p:
        y = p.y
        if discovered[y] is False:
            parents[y] = v
            print(f'Processing ({v}, {y})')
            process_edge(v, y, parents)
            DFS2(graph, y)
        elif not processed[y] or graph.directed:
            print(f'Processing ({v}, {y})')
            process_edge(v, y, parents)
        p = p.next
    print(f'Exiting {v}')
    time += 1
    processed[v] = True


# print('----')
# DFS2(graph2, 0)
