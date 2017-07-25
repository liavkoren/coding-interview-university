import matplotlib.pyplot as plt
import numpy as np

import adjacency_list


class GraphInteractor:
    epsilon = 0.00005  # max pixel distance to count as a vertex hit

    def __init__(self, ax, graph):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.graph = graph
        self._ind = None
        self.lines = []
        self.Xs = np.random.random(self.graph.nnodes)
        self.Ys = np.random.random(self.graph.nnodes)
        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.configure_display()
        self.render_graph()
        print(f'Xs: {self.Xs}, \nYs: {self.Ys}')

    def configure_display(self):
        for name in ['left', 'right', 'top', 'bottom']:
            spine = self.ax.spines[name]
            spine.set_visible(False)

    def render_graph(self):
        self.ax.scatter(self.Xs, self.Ys)
        for index, node in enumerate(self.graph.nodes):
            for adjacent_node in node:
                x = [self.Xs[index], self.Xs[adjacent_node.y]]
                y = [self.Ys[index], self.Ys[adjacent_node.y]]
                self.ax.plot(x, y)
                # self.lines.append(ax.plot(x, y, animated=True))
        plt.show()

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.canvas.blit(self.ax.bbox)

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        d = (self.Xs - event.xdata)**2 + (self.Ys - event.ydata)**2
        index = d.argmin()
        print(f'distances: {d}')
        if d[index] >= self.epsilon:
            return None
        print(f'found {index}')
        return index

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None

        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        # x, y = event.xdata, event.ydata
        self.Xs[self._ind], self.Ys[self._ind] = event.xdata, event.ydata
        self.render_graph()

        '''
        for line in self.lines[0]:  # why is this 2d nested??
            self.ax.draw_artist(line)
        self.canvas.restore_region(self.background)
        '''
        '''
        vertices = self.pathpatch.get_path().vertices

        vertices[self._ind] = x, y
        # self.line.set_data(zip(*vertices))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.pathpatch)
        # self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        '''


fig, ax = plt.subplots()
interactor = GraphInteractor(graph=adjacency_list.articulated_graph, ax=ax)
