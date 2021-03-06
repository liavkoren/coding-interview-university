"""
Based on Skiena's discussion of Heaps in Algorithm Design Manual, pp 108 - 118.

Skiena uses 1-based indexing. 0-based indexing is: for level i, items range from
[2^(i-1) - 1 , 2^i - 1], ie:
      0
    1   2
   3 4 5 6
  7 8 ... 15

Also uses the lovely anytree library to provide ascii tree printing. Still need
to debug rendering to dot file.
"""
import reprlib

from anytree import (
    Node,
    RenderTree,
)
from anytree.render import ContStyle


def heapsort(items):
    for index, item in enumerate(Heap(items)):
        items[index] = item
    return items


class HeapError(Exception):
    pass


class Heap:
    """ It's a min heap. Takes a list of stuff. """
    def __init__(self, *args):
        self.items = []
        if args and isinstance(args[0], list):
            if len(args) > 1:
                raise TypeError('Heap either takes a list or *args, not both.')
            data = args[0]
        else:
            data = args
        for item in data:
            self.insert(item)

    def __getitem__(self, index):
        # poke around in the heap at your own peril.
        return self.items[index]

    def __iter__(self):
        # Implementing getitem means we have to implement this, otherwise
        # iteration over the heap isn't deterministic. Perhaps can use Skiena's
        # heap_compare algorithm to progressively find each kth biggest item, to
        # do a better, non-destructive version of this?
        try:
            while True:
                yield self.extract()
        except HeapError:
            pass

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f'Heap({reprlib.repr(self.items)})'

    def __setitem__(self, index, value):
        self.items[index] = value

    def __str__(self):
        if not self.items:
            return ''
        out = []
        for prefix, _, node in self.to_tree():
            out.append(f'{prefix}{node.name}')
        return '\n'.join(out)

    def _swap(self, one, two):
        self[one], self[two] = self[two], self[one]

    def _bubble_up(self, index):
        parent_index = self.parent(index)
        if parent_index is None:
            return
        if self[parent_index] > self[index]:
            self._swap(index, parent_index)
            self._bubble_up(parent_index)

    def _bubble_down(self, item_index):
        youngest_child_index = self.youngest_child(item_index)
        min_index = item_index
        max_heap_index = len(self) - 1
        for j in [0, 1]:
            if youngest_child_index + j <= max_heap_index:
                if self[min_index] > self[youngest_child_index + j]:
                    min_index = youngest_child_index + j
        if min_index != item_index:
            self._swap(item_index, min_index)
            self._bubble_down(min_index)

    # def render(self, filename):
    #    RenderTreeGraph(self.to_tree()).to_picture(filename)

    def to_tree(self):
        nodes = []
        for index, item in enumerate(self):
            node = Node(name=item)
            nodes.append(node)
            if index > 0:
                parent_index = self.parent(index)
                node.parent = nodes[parent_index]
        return RenderTree(nodes[0], style=ContStyle())

    def insert(self, item):
        self.items.append(item)
        self._bubble_up(len(self)-1)

    def extract(self):
        if len(self) > 0:
            self._swap(0, len(self)-1)
            out = self.items.pop()
            self._bubble_down(0)
            return out
        else:
            raise HeapError('Heap is empty')

    def peek(self):
        return self[0]

    @staticmethod
    def parent(n):
        if n == 0:
            return None
        return int((n-1)/2)

    @staticmethod
    def youngest_child(n):
        return 2*n + 1

    @property
    def stream(self):
        """ Provides an iterable interface to the heap. Consuming the stream exhausts the heap. """
        try:
            while True:
                yield self.extract()
        except HeapError:
            pass
