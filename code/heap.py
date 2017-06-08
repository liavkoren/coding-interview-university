# from anytree import Node, RenderTree

'''
Addressing scheme using zero indexing is for level i, items range from
[2^i-1, 2^i-2] inclusive.
'''
from anytree import (
    Node,
    RenderTree,
)
from anytree.dotexport import RenderTreeGraph
from anytree.render import ContStyle


class Heap:
    """ It's a min heap. Takes a list of stuff. """
    def __init__(self, *args):
        self.items = []
        self.pprint = []

    def __str__(self):
        if not self.items:
            return ''
        out = []
        for prefix, _, node in self.to_tree():
            out.append(f'{prefix}{node.name}')
        return '\n'.join(out)

    def render(self, filename):
        RenderTreeGraph(self.to_tree()).to_picture(filename)

    def to_tree(self):
        nodes = []
        for index, item in enumerate(self.items):
            node = Node(name=item)
            nodes.append(node)
            if index > 0:
                parent_index = self.parent(index)
                node.parent = nodes[parent_index]
        return RenderTree(nodes[0], style=ContStyle())

    def _sift_down(self):
        # needed for extract_max
        pass

    def _swap(self, first, second):
        self.items[first], self.items[second] = self.items[second], self.items[first]

    def _bubble_up(self, index):
        parent_index = self.parent(index)
        if not parent_index:
            return
        if self.items[parent_index] > self.items[index]:
            self._swap(index, parent_index)
            self._bubble_up(parent_index)

    def _heapify(self):
        # create a heap from an array of elements, needed for heap_sort
        pass

    def insert(self, item):
        self.items.append(item)
        self._bubble_up(self.size()-1)

    def extract(self):
        # returns the max item, removing it
        pass

    def parent(self, n):
        if n == 0:
            return None
        return int((n-1)/2)

    def youngest_child(self, n):
        return 2*n + 1

    # ------
    def heap_sort(self):
        # take an unsorted array and turn it into a sorted array in-place using a max heap
        pass

    def peek(self):
        # returns the max item, without removing it
        pass

    def size(self):
        return len(self.items)

    def is_empty(self):
        # returns true if heap contains no elements
        pass

    def remove(self):
        # removes item at index x
        pass

# note: using a min heap instead would save operations, but double the space needed (cannot do in-place).


