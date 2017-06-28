from collections import deque
from enum import Enum

import attr
from attr.validators import instance_of


class GraphException(Exception):
    pass


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
    directed = attr.ib(default=False)
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

    def add_edge(self, x, y, weight=1):
        # Lesson:
        # We need this inner because we can't cleanly recurse on an instance's
        # method while using an instance attribute as the bail-out condition,
        # and also leave the attribute immutable. Ie: 'directed' gets toggled
        # for the two calls to inner, but self.directed should be immutable
        # for the instance.
        def inner(x, y, weight, directed=False):
            while max(x, y) + 1 > len(self.nodes):
                self.nodes.append(None)
            edgenode = EdgeNode(y=y, weight=weight, next=self.nodes[x])
            self.nodes[x] = edgenode
            if directed:
                self.nedges += 1
            else:
                inner(y, x, weight, directed=True)
        inner(x, y, weight=1, directed=self.directed)


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


def find_cycles(search_state, x, y):
    print(f'    Processing edge ({x:02}, {y:02})')
    # Why do I need to do this check aginst (x,y) / (y, x) when Skiena doesn't include it?
    if search_state.node_parents[x] == y or search_state.node_parents[y] == x:
        return
    print(f'    Found cycle from {x} to {y}: ')
    find_path(x, y, search_state.node_parents)


def find_path(src, dst, parent_list):
    if src == dst or dst is None:
        pass
    else:
        # Do not ignore the order of recursive calls w/r/t other code! Code after
        # a recursive call is deferred.
        find_path(parent_list[src], dst, parent_list)
        print(f'\t{parent_list[src]}')


class NodeStates(Enum):
    undiscovered = 0
    discovered = 1
    processed = 2


class EdgeTypes(Enum):
    tree = 0
    back = 1
    forward = 2
    cross = 3


def edge_classification(x, y, search_state):
    if search_state.node_parents[y] == x:
        return EdgeTypes.tree
    if search_state.node_states[y] is NodeStates.discovered:
        return EdgeTypes.back
    if search_state.node_states[y] is NodeStates.processed and search_state.entry_times[y] > search_state.entry_times[x]:
        return EdgeTypes.forward
    if search_state.node_states[y] is NodeStates.processed and search_state.entry_times[y] < search_state.entry_times[x]:
        return EdgeTypes.cross
    raise GraphException('Unclassifiable edge.')


@attr.s
class SearchState:
    """ Keeps track of all the book keeping while doing a graph traversal. """
    node_states = attr.ib(default=None)
    node_parents = attr.ib(default=None)
    entry_times = attr.ib(default=None)
    exit_times = attr.ib(default=None)
    reachable_ancestors = attr.ib(default=None)
    out_degrees = attr.ib(default=None)
    graph = attr.ib(instance_of(Graph))

    def __str__(self):
        stats = [
            f'\tParents: {self.node_parents}',
            f'\tEntry times: {self.entry_times}',
            f'\tExit times: {self.exit_times}',
            f'\tEarliest ancestors: {self.reachable_ancestors}',
            f'\tOut degrees: {self.out_degrees}',
        ]
        return '\n'.join(stats)

    def __attrs_post_init__(self):
        self.node_states = [NodeStates.undiscovered] * self.graph.nnodes
        self.node_parents = [None] * self.graph.nnodes
        self.entry_times = [0] * self.graph.nnodes
        self.exit_times = [0] * self.graph.nnodes
        self.out_degrees = [0] * self.graph.nnodes
        self.reachable_ancestors = [0] * self.graph.nnodes


def DFS(graph, start, process_node_early, process_node_late, process_edge):
    time = 0
    search_state = SearchState(graph=graph)

    def recurse(v):
        nonlocal time
        time += 1
        search_state.node_states[v] = NodeStates.discovered
        search_state.entry_times[v] = time
        process_node_early(search_state, v)
        adjacent_nodes = graph.nodes[v]
        # Needs to handle disconnected singletons
        for node in adjacent_nodes:
            if search_state.node_states[node.y] is NodeStates.undiscovered:
                search_state.node_parents[node.y] = v
                process_edge(search_state, v, node.y)  # <-- falls over at edge (1, 4)
                recurse(node.y)
            elif search_state.node_states[node.y] is NodeStates.discovered or graph.directed:
                process_edge(search_state, v, node.y)
        process_node_late(search_state, v)
        time += 1
        search_state.exit_times[v] = time
        search_state.node_states[v] = NodeStates.processed

    recurse(start)
    print(f'Search state:\n{search_state}')


def process_node_early(search_state, index):
    print(f'Entering node #{index:02}')
    search_state.reachable_ancestors[index] = index


def process_node_late(search_state, index):
    # Are we at the root?
    if search_state.node_parents[index] is None:
        if search_state.out_degrees[index] > 1:
            print(f'Root articulation vertex: {index}')
            return
    # is the parent the root?
    root = search_state.node_parents[search_state.node_parents[index]] is None
    if search_state.reachable_ancestors[index] == search_state.node_parents[index] and not root:
        print(f'Parent articulation vertex: {search_state.node_parents[index]} for {index}')
        return

    if search_state.reachable_ancestors[index] == index:
        print(f'Bridge articulation vertex: {search_state.node_parents[index]} for {index}')
        # check that it's not a leaf:
        if search_state.out_degrees[index] > 0:
            print(f'Bridge articulation vertex: {index}')
            return

    # If the child is roped back up to a higher part of the graph than its parent
    # we can bubble that older ancestor up to the parent:
    node_ancestor_index = search_state.reachable_ancestors[index]
    node_entry_time = search_state.entry_times[node_ancestor_index]

    parent_ancestor_index = search_state.reachable_ancestors[search_state.node_parents[index]]
    parent_entry_time = search_state.entry_times[parent_ancestor_index]
    if node_entry_time < parent_entry_time:
        parent_index = search_state.node_parents[index]
        search_state.reachable_ancestors[parent_index] = node_ancestor_index
    print(f'Leaving node #{index:02}')


def find_ancestors(search_state, x, y):
    edge_class = edge_classification(x, y, search_state)
    if edge_class is EdgeTypes.tree:
        search_state.out_degrees[x] += 1  # <--
    if edge_class is EdgeTypes.back and search_state.node_parents[x] != y:
        ancestor_index = search_state.reachable_ancestors[x]
        found_older_ancestor = search_state.entry_times[y] < search_state.entry_times[ancestor_index]
        if found_older_ancestor:
            search_state.reachable_ancestors[x] = y


# DFS(graph2, 0, process_node_early, process_node_late, find_ancestors)

articulated_graph = Graph()
articulated_graph.add_edge(0, 5)
articulated_graph.add_edge(0, 1)
articulated_graph.add_edge(4, 1)
articulated_graph.add_edge(1, 2)
articulated_graph.add_edge(2, 3)
articulated_graph.add_edge(3, 4)

print(articulated_graph)
print('-----')
DFS(articulated_graph, 0, process_node_early, process_node_late, find_ancestors)

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
            # process_edge(v, y, parents)
            DFS2(graph, y)
        elif not processed[y] or graph.directed:
            print(f'Processing ({v}, {y})')
            # process_edge(v, y, parents)
        p = p.next
    print(f'Exiting {v}')
    time += 1
    processed[v] = True


# print('----')
# DFS2(graph2, 0)


# --------
# DAGs:

dag = Graph(directed=True)
dag.add_edge(0, 1)
dag.add_edge(0, 4)
dag.add_edge(1, 2)
dag.add_edge(1, 3)
dag.add_edge(2, 3)
dag.add_edge(2, 6)
dag.add_edge(3, 4)
dag.add_edge(3, 5)
dag.add_edge(4, 5)
dag.add_edge(5, 6)

assert dag.nnodes == 7
assert dag.nedges == 10
print('---')
print(dag)
