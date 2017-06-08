'''
Based on Skiena's discussion of Heaps in Algorithm Design Manual, pp 108 - 118.

Skiena uses 1-based indexing. 0-based indexing is: for level i, items range from
[2^i-1, 2^i-2].

Also uses the lovely anytree library to provide ascii tree printing. Still need
to debug rendering to dot file.
'''
from anytree import (
    Node,
    RenderTree,
)
from anytree.dotexport import RenderTreeGraph
from anytree.render import ContStyle


class HeapError(Exception):
    pass


class Heap:
    """ It's a min heap. Takes a list of stuff. """
    def __init__(self, *args):
        self.items = []
        for item in args:
            self.insert(item)

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
        if parent_index is None:
            return
        if self.items[parent_index] > self.items[index]:
            self._swap(index, parent_index)
            self._bubble_up(parent_index)

    def _bubble_down(self, item_index):
        youngest_child_index = self.youngest_child(item_index)
        min_index = item_index
        max_heap_index = self.size() - 1
        for j in [0, 1]:
            if youngest_child_index + j <= max_heap_index:
                if self.items[min_index] > self.items[youngest_child_index + j]:
                    min_index = youngest_child_index + j
        if min_index != item_index:
            self._swap(item_index, min_index)
            self._bubble_down(min_index)

    def _heapify(self):
        # create a heap from an array of elements, needed for heap_sort
        pass

    def insert(self, item):
        self.items.append(item)
        self._bubble_up(self.size()-1)

    def extract(self):
        if self.size() > 0:
            self._swap(0, self.size()-1)
            out = self.items.pop()
            self._bubble_down(0)
            return out
        else:
            raise HeapError('Heap is empty')

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


