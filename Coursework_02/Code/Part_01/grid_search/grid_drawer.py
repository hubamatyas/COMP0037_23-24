from typing import Optional

import locale
import ghostscript  # type: ignore
import pyscreenshot as ImageGrab

from .grid import Grid

from .graphics import *
from .graphics import _root

import numpy as np
import matplotlib.pyplot as plt


class GridDrawer(object):

    def __init__(self, grid: Grid, maximum_grid_drawer_window_height_in_pixels: int,
                 top_left_in_pixels: Optional[int] = None):

        self._grid = grid
        width = grid.width()
        height = grid.height()

        # Make sure that the height of the window is less than the specified maximum
        self._cell_size = max(
            10, maximum_grid_drawer_window_height_in_pixels / height)

        # Create the window
        pixel_width = width * self._cell_size
        pixel_height = height * self._cell_size

        self._win = GraphWin(grid.name(), pixel_width,
                             pixel_height, autoflush=False)

        # If the x and y coordinates are specified, then set the geometry; this is a bit of a hack
        # Note we have to repeat type: ignore so mypy ignores both lines
        if top_left_in_pixels is not None:
            self._win.master.geometry('%dx%d+%d+%d' % (pixel_width, pixel_height,  # type: ignore
                                      top_left_in_pixels[0], top_left_in_pixels[1]))  # type: ignore

        # Allocate the cells
        self._rectangles = [[Rectangle(Point(i * self._cell_size, (height - j - 1) * self._cell_size),
                                       Point((i+1) * self._cell_size, (height - j) * self._cell_size))
                            for i in range(width)]
                            for j in range(height)]

        for i in range(width):
            for j in range(height):
                self._rectangles[j][i].draw(self._win)

    def reset(self):
        pass

    # Save the window
    def save_screenshot(self, filename: str):

        # We store two versions. The first saves a pdf, the
        # second supports all other types. pdf is recommended because
        # the image is of much higher quality.

        print(f"Saving file {filename}")

        if filename.endswith("pdf"):

            # From https://pypi.org/project/ghostscript/
            # and https://stackoverflow.com/questions/57787990/

            # Save the file as postscript
            self._win.postscript(file="tmp.ps", colormode="color")

            # Set up the arguments and call ghostscript to
            # do the conversion
            args = [
                "ps2pdf",  # actual value doesn't matter
                "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dEPSCrop",
                "-sDEVICE=pdfwrite",
                "-sOutputFile=" + filename,
                "-f",  "tmp.ps"
            ]

            encoding = locale.getpreferredencoding()
            encoded_args = [a.encode(encoding) for a in args]

            ghostscript.Ghostscript(*encoded_args)

            # Delete the temporary file
            os.remove("tmp.ps")

        else:

            # From https://stackoverflow.com/questions/66672786
            x = self._win.winfo_rootx()
            y = self._win.winfo_rooty()
            x1 = x+self._win.winfo_width()
            y1 = y+self._win.winfo_height()
            screenshot_rgba = ImageGrab.grab().crop((x, y, x1, y1))
            screenshot_rgb = screenshot_rgba.convert("RGB")
            screenshot_rgb.save(filename)

        print(f"Finished saving {filename}")

    def fancy_save_screenshot(self, filename: str, is_showing: bool = False, title: str = None):
        value_function = np.rot90(self._grid._values, k=1)

        # Set up heatmap and default value for nans
        cmap = plt.cm.get_cmap('viridis', 256)
        cmap.set_bad(color='white')

        # Plot the value function
        plt.imshow(value_function, cmap=cmap, interpolation='nearest')

        # Set the title
        if title is not None:
            plt.title(title)

        # Formatting
        # Set figure size to match the dimensions of the value function
        plt.gcf().set_size_inches(value_function.shape[1] / 2, value_function.shape[0] / 2)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        # Remove the border around the heatmap
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['bottom'].set_visible(False)
        # plt.gca().spines['left'].set_visible(False)

        # Show the values within each cell and add border around each cell
        for i in range(value_function.shape[0]):
            for j in range(value_function.shape[1]):
                plt.text(j, i, f"{value_function[i, j]:.2f}", ha='center', va='center', color='black')
                plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black'))

        # Remove margin and padding
        plt.margins(0)
        plt.gcf().subplots_adjust(bottom=0, top=1, left=0, right=1)

        if is_showing:
            plt.show()

        plt.savefig(filename)
        plt.close()

    def update(self):
        raise NotImplementedError()

    def wait_for_key_press(self):
        self._win.getKey()
