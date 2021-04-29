import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.__version__, matplotlib.get_backend())


class SliceViewer(object):
    def __init__(self, ax, X, aspect='auto'):
        self.ax = ax
        #ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.mode = 0
        self.rows = X.shape[0]
        self.cols = X.shape[1]
        self.slices = X.shape[2]
        self.ind = self.slices // 2
        self.index = (slice(None), self.ind, slice(None))

        self.im = ax.imshow(self.X[self.index], aspect=aspect)
        #self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def viewmode(self):
        mode = self.mode
        if mode == 0: # XY-plane
            self.rows = self.X.shape[0]
            self.cols = self.X.shape[1]
            self.slices = self.X.shape[2]
            self.index = (slice(None), slice(None), self.ind)
        elif mode == 1: # XZ-plane
            self.rows = self.X.shape[0]
            self.cols = self.X.shape[2]
            self.slices = self.X.shape[1]
            self.index = (slice(None), self.ind, slice(None))
        elif mode == 2: # YZ-plane
            self.rows = self.X.shape[1]
            self.cols = self.X.shape[2]
            self.slices = self.X.shape[0]
            self.index = (self.ind, slice(None), slice(None))
        else:
            raise KeyError('invalid view mode.')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def click_view(self, event):
        mode = self.mode
        if event.key == '#':
            self.mode = (mode + 1) % 3
            self.viewmode()
        self.update()

    def update(self):
        mode = self.mode
        if mode == 0:  # XY-plane
            self.rows = self.X.shape[0]
            self.cols = self.X.shape[1]
            self.slices = self.X.shape[2]
            self.index = (slice(None), slice(None), self.ind)
            self.ax.set_ylabel('slice %s (XY-plane)' % self.ind)
        elif mode == 1:  # XZ-plane
            self.rows = self.X.shape[0]
            self.cols = self.X.shape[2]
            self.slices = self.X.shape[1]
            self.index = (slice(None), self.ind, slice(None))
            self.ax.set_ylabel('slice %s (XZ-plane)' % self.ind)
        elif mode == 2:  # YZ-plane
            self.rows = self.X.shape[1]
            self.cols = self.X.shape[2]
            self.slices = self.X.shape[0]
            self.index = (self.ind, slice(None), slice(None))
            self.ax.set_ylabel('slice %s (YZ-plane)' % self.ind)
        else:
            raise KeyError('invalid view mode.')
        self.im.set_data(self.X[self.index])
        #self.im.set_data(self.X[:, :, self.ind])

        self.im.axes.figure.canvas.draw()

    def plot(volume1, volume2, volume3, volume4, volume5, volume6):
        fig, ax = plt.subplots(2, 3, figsize=(12, 9))
        viewer1 = SliceViewer(ax[0, 0], volume1)
        viewer2 = SliceViewer(ax[0, 1], volume2)
        viewer3 = SliceViewer(ax[0, 2], volume3)
        viewer4 = SliceViewer(ax[1, 0], volume4)
        viewer5 = SliceViewer(ax[1, 1], volume5)
        viewer6 = SliceViewer(ax[1, 2], volume6)
        fig.canvas.mpl_connect('scroll_event', viewer1.onscroll)
        fig.canvas.mpl_connect('scroll_event', viewer2.onscroll)
        fig.canvas.mpl_connect('scroll_event', viewer3.onscroll)
        fig.canvas.mpl_connect('scroll_event', viewer4.onscroll)
        fig.canvas.mpl_connect('scroll_event', viewer5.onscroll)
        fig.canvas.mpl_connect('scroll_event', viewer6.onscroll)
        fig.canvas.mpl_connect('key_press_event', viewer1.click_view)
        fig.canvas.mpl_connect('key_press_event', viewer2.click_view)
        fig.canvas.mpl_connect('key_press_event', viewer3.click_view)
        fig.canvas.mpl_connect('key_press_event', viewer4.click_view)
        fig.canvas.mpl_connect('key_press_event', viewer5.click_view)
        fig.canvas.mpl_connect('key_press_event', viewer6.click_view)
        plt.show()
