import matplotlib.pyplot as plt
import pdb
import matplotlib
print(matplotlib.__version__, matplotlib.get_backend())

class SliceViewer(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.mode = 1
        self.rows, self.cols, self.slices, _ = X.shape
        self.ind = self.slices // 2
        self.index = (slice(None), slice(None), self.ind)

        self.im = ax.imshow(self.X[self.index])
        self.update()

    def viewmode(self):
        mode = self.mode
        if mode == 1: # XY-plane
            self.rows, self.cols, self.slices, _ = X.shape
            self.index = (slice(None), slice(None), self.ind)
        elif mode == 2: # XZ-plane
            self.rows, self.slices, self.cols, _ = X.shape
            self.index = (slice(None), self.ind, slice(None))
        elif mode == 3: # YZ-plane
            self.slices, self.rows, self.cols, _ = X.shape
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
        pdb.set_trace()
        if event.button == '#':
            self.mode += 1
            self.viewmode()
        self.update()

    def update(self):
        self.im.set_data(self.X[self.index])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def plot(volume):
        fig, ax = plt.subplots()
        viewer = SliceViewer(ax, volume)
        #fig.canvas.mpl_connect('scroll_event', viewer.onscroll)
        fig.canvas.mpl_connect('click_# event', viewer.click_view)
        plt.show()