import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Based on: https://github.com/pyxem/pyxem/blob/master/pyxem/utils/peakfinder2D_gui.py

class PeakExplorerBase:

    def __init__(self):
        self.signal = None
        self.indices = None

    def interactive(self, signal):
        self.signal = signal
        self.indices = self.signal.axes_manager.indices
        self.init_ui()

    def init_ui(self):
        raise NotImplementedError

    def get_data(self):
        _slices = self.signal._get_array_slices(self.indices, isNavigation=True)
        return self.signal.data[_slices]

    def get_peaks(self):
        shape = self.get_data().shape
        index = self.indices[0]
        ori = self.orientations.data[index][0] # use 2d indexer?
        shape = self.get_data().shape
        peaks = self.indexer.get_indices(ori, shape)
        return peaks.take([6,7,5], axis=1)


class PeakExplorer(PeakExplorerBase):
    """
    Explore peaks using a Jupyter notebook-based user interface
    """

    def __init__(self, indexer, orientations, imshow_kwargs={}):
        super(PeakExplorer, self).__init__()
        self.ax = None
        self.image = None
        self.pts = None
        self.param_container = None
        self.imshow_kwargs = imshow_kwargs
        self.orientations = orientations
        self.indexer = indexer
        
    def init_ui(self):
        self.create_navigator()
        self.plot()

    def create_navigator(self):
        from ipywidgets import HBox
        container = HBox()
        if self.signal.axes_manager.navigation_dimension == 2:
            container = self.create_navigator_2d()
        elif self.signal.axes_manager.navigation_dimension == 1:
            container = self.create_navigator_1d()
        display(container)
    
    def create_navigator_1d(self):
        import ipywidgets as ipyw
        x_min, x_max = 0, self.signal.axes_manager.navigation_size - 1
        x_text = ipyw.BoundedIntText(value=self.indices[0],
                                     description="Index:", min=x_min,
                                     max=x_max,
                                     layout=ipyw.Layout(flex='0 1 auto',
                                                        width='auto'))
        randomize = ipyw.Button(description="Randomize",
                                layout=ipyw.Layout(flex='0 1 auto',
                                                   width='auto'))
        container = ipyw.HBox((x_text, randomize))

        def on_index_change(change):
            self.indices = (x_text.value,)
            self.replot_image()

        def on_randomize(change):
            from random import randint
            x = randint(x_min, x_max)
            x_text.value = x

        x_text.observe(on_index_change, names='value')
        randomize.on_click(on_randomize)
        return container

    def create_navigator_2d(self):
        import ipywidgets as ipyw
        x_min, y_min = 0, 0
        x_max, y_max = self.signal.axes_manager.navigation_shape
        x_max -= 1
        y_max -= 1
        x_text = ipyw.BoundedIntText(value=self.indices[0], description="x",
                                     min=x_min, max=x_max,
                                     layout=ipyw.Layout(flex='0 1 auto',
                                                        width='auto'))
        y_text = ipyw.BoundedIntText(value=self.indices[1], description="y",
                                     min=y_min, max=y_max,
                                     layout=ipyw.Layout(flex='0 1 auto',
                                                        width='auto'))
        randomize = ipyw.Button(description="Randomize",
                                layout=ipyw.Layout(flex='0 1 auto',
                                                   width='auto'))
        container = ipyw.HBox((x_text, y_text, randomize))

        def on_index_change(change):
            self.indices = (x_text.value, y_text.value)
            self.replot_image()

        def on_randomize(change):
            from random import randint
            x = randint(x_min, x_max)
            y = randint(y_min, y_max)
            x_text.value = x
            y_text.value = y

        x_text.observe(on_index_change, names='value')
        y_text.observe(on_index_change, names='value')
        randomize.on_click(on_randomize)
        return container

    def plot(self):
        self.ax = None
        self.plot_image()
        self.set_title()
        self.plot_peaks()

    def plot_image(self):
        if self.ax is None:
            self.ax = plt.figure().add_subplot(111)
        z = self.get_data()
        self.image = self.ax.imshow(z, **self.imshow_kwargs)
        self.ax.set_xlim(0, z.shape[0])
        self.ax.set_ylim(0, z.shape[1])
        plt.show()

    def replot_image(self):
        if not plt.get_fignums():
            self.plot()
        z = self.get_data()
        self.image.set_data(z)
        self.replot_peaks()
        self.set_title()
        plt.draw()

    def plot_peaks(self):
        peaks = self.get_peaks()
        self.pts = self.ax.scatter(peaks[:, 1], peaks[:, 0], c=peaks[:, 2], marker="+", cmap="viridis")
        plt.show()

    def replot_peaks(self):
        if not plt.get_fignums():
            self.plot()
        peaks = self.get_peaks()
        self.pts.remove()
        self.pts = self.ax.scatter(peaks[:, 1], peaks[:, 0], c=peaks[:, 2], marker="+", cmap="viridis")
        plt.draw()
    
    def set_title(self):
        shape = self.get_data().shape
        index = self.indices[0]
        ori = self.orientations.data[index][0] # use 2d indexer?
        n = ori.number
        center_x = ori.center_x
        center_y = ori.center_y
        scale = ori.scale
        alpha = ori.alpha
        beta = ori.beta
        gamma = ori.gamma
        score = ori.score
        phase = ori.phase
        title = "Image #{}/{}".format(index, len(self.signal))
        self.ax.set_title("{}\nal: {:.2f} | be: {:.2f} | ga: {:.2f} | scale = {:.1f}\nscore = {:.1f} | proj = {} | phase = {}".format(title, alpha, beta, gamma, scale, score, n, phase))
  