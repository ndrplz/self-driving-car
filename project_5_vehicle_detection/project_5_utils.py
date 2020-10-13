import os
import sys
from os.path import exists
from os.path import join

import cv2
import numpy as np


def get_file_list_recursively(top_directory):
    """
    Get list of full paths of all files found under root directory "top_directory".
    If a list of allowed file extensions is provided, files are filtered according to this list.

    Parameters
    ----------
    top_directory: str
        Root of the hierarchy

    Returns
    -------
    file_list: list
        List of files found under top_directory (with full path)
    """
    if not exists(top_directory):
        raise ValueError('Directory "{}" does NOT exist.'.format(top_directory))

    file_list = []

    for cur_dir, cur_subdirs, cur_files in os.walk(top_directory):

        for file in cur_files:
            file_list.append(join(cur_dir, file))
            sys.stdout.write(
                '\r[{}] - found {:06d} files...'.format(top_directory, len(file_list)))
            sys.stdout.flush()

    sys.stdout.write(' Done.\n')

    return file_list


def stitch_together(input_images, layout, resize_dim=None, off_x=None, off_y=None,
                    bg_color=(0, 0, 0)):
    """
    Stitch together N input images into a bigger frame, using a grid layout.
    Input images can be either color or grayscale, but must all have the same size.

    Parameters
    ----------
    input_images : list
        List of input images
    layout : tuple
        Grid layout of the stitch expressed as (rows, cols)
    resize_dim : couple
        If not None, stitch is resized to this size
    off_x : int
        Offset between stitched images along x axis
    off_y : int
        Offset between stitched images along y axis
    bg_color : tuple
        Color used for background

    Returns
    -------
    stitch : ndarray
        Stitch of input images
    """

    if len(set([img.shape for img in input_images])) > 1:
        raise ValueError('All images must have the same shape')

    if len(set([img.dtype for img in input_images])) > 1:
        raise ValueError('All images must have the same data type')

    # determine if input images are color (3 channels) or grayscale (single channel)
    if len(input_images[0].shape) == 2:
        mode = 'grayscale'
        img_h, img_w = input_images[0].shape
    elif len(input_images[0].shape) == 3:
        mode = 'color'
        img_h, img_w, img_c = input_images[0].shape
    else:
        raise ValueError('Unknown shape for input images')

    # if no offset is provided, set to 10% of image size
    if off_x is None:
        off_x = img_w // 10
    if off_y is None:
        off_y = img_h // 10

    # create stitch mask
    rows, cols = layout
    stitch_h = rows * img_h + (rows + 1) * off_y
    stitch_w = cols * img_w + (cols + 1) * off_x
    if mode == 'color':
        bg_color = np.array(bg_color)[None, None, :]  # cast to ndarray add singleton dimensions
        stitch = np.uint8(np.repeat(np.repeat(bg_color, stitch_h, axis=0), stitch_w, axis=1))
    elif mode == 'grayscale':
        stitch = np.zeros(shape=(stitch_h, stitch_w), dtype=np.uint8)

    for r in range(0, rows):
        for c in range(0, cols):

            list_idx = r * cols + c

            if list_idx < len(input_images):
                if mode == 'color':
                    stitch[r * (off_y + img_h) + off_y: r * (off_y + img_h) + off_y + img_h,
                    c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w,
                    :] = input_images[list_idx]
                elif mode == 'grayscale':
                    stitch[r * (off_y + img_h) + off_y: r * (off_y + img_h) + off_y + img_h,
                    c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w] \
                        = input_images[list_idx]

    if resize_dim:
        stitch = cv2.resize(stitch, dsize=(resize_dim[::-1]))

    return stitch


class Rectangle:
    """
    2D Rectangle defined by top-left and bottom-right corners.
    Parameters
    ----------
    x_min : int
        x coordinate of top-left corner.
    y_min : int
        y coordinate of top-left corner.
    x_max : int
        x coordinate of bottom-right corner.
    y_min : int
        y coordinate of bottom-right corner.
    """

    def __init__(self, x_min, y_min, x_max, y_max, label=""):

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_side = self.x_max - self.x_min
        self.y_side = self.y_max - self.y_min

        self.label = label

    def intersect_with(self, rect):
        """
        Compute the intersection between this instance and another Rectangle.

        Parameters
        ----------
        rect : Rectangle
            The instance of the second Rectangle.

        Returns
        -------
        intersection_area : float
            Area of intersection between the two rectangles expressed in number of pixels.
        """
        if not isinstance(rect, Rectangle):
            raise ValueError('Cannot compute intersection if "rect" is not a Rectangle')

        dx = min(self.x_max, rect.x_max) - max(self.x_min, rect.x_min)
        dy = min(self.y_max, rect.y_max) - max(self.y_min, rect.y_min)

        if dx >= 0 and dy >= 0:
            intersection = dx * dy
        else:
            intersection = 0.

        return intersection

    def resize_sides(self, ratio, bounds=None):
        """
        Resize the sides of rectangle while mantaining the aspect ratio and center position.
        Parameters
        ----------
        ratio : float
            Ratio of the resize in range (0, infinity), where 2 means double the size and 0.5 is half of the size.
        bounds: tuple, optional
            If present, clip the Rectangle to these bounds=(xbmin, ybmin, xbmax, ybmax).
        Returns
        -------
        rectangle : Rectangle
            Reshaped Rectangle.
        """

        # compute offset
        off_x = abs(ratio * self.x_side - self.x_side) / 2
        off_y = abs(ratio * self.y_side - self.y_side) / 2

        # offset changes sign according if the resize is either positive or negative
        sign = np.sign(ratio - 1.)
        off_x = np.int32(off_x * sign)
        off_y = np.int32(off_y * sign)

        # update top-left and bottom-right coords
        new_x_min, new_y_min = self.x_min - off_x, self.y_min - off_y
        new_x_max, new_y_max = self.x_max + off_x, self.y_max + off_y

        # eventually clip the coordinates according to the given bounds
        if bounds:
            b_x_min, b_y_min, b_x_max, b_y_max = bounds
            new_x_min = max(new_x_min, b_x_min)
            new_y_min = max(new_y_min, b_y_min)
            new_x_max = min(new_x_max, b_x_max)
            new_y_max = min(new_y_max, b_y_max)

        return Rectangle(new_x_min, new_y_min, new_x_max, new_y_max)

    def draw(self, frame, color=255, thickness=2, draw_label=False):
        """
        Draw Rectangle on a given frame.
        Notice: while this function does not return anything, original image `frame` is modified.
        Parameters
        ----------
        frame : 2D / 3D np.array
            The image on which the rectangle is drawn.
        color : tuple, optional
            Color used to draw the rectangle (default = 255)
        thickness : int, optional
            Line thickness used t draw the rectangle (default = 1)
        draw_label : bool, optional
            If True and the Rectangle has a label, draws it on the top of the rectangle.
        Returns
        -------
        None
        """
        if draw_label and self.label:
            # compute text size
            text_font, text_scale, text_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (text_w, text_h), baseline = cv2.getTextSize(self.label, text_font, text_scale,
                                                         text_thick)

            # draw rectangle on which text will be displayed
            text_rect_w = min(text_w, self.x_side - 2 * baseline)
            out = cv2.rectangle(frame.copy(), pt1=(self.x_min, self.y_min - text_h - 2 * baseline),
                                pt2=(self.x_min + text_rect_w + 2 * baseline, self.y_min),
                                color=color, thickness=cv2.FILLED)
            cv2.addWeighted(frame, 0.75, out, 0.25, 0, dst=frame)

            # actually write text label
            cv2.putText(frame, self.label, (self.x_min + baseline, self.y_min - baseline),
                        text_font, text_scale, (0, 0, 0), text_thick, cv2.LINE_AA)

            # add text rectangle border
            cv2.rectangle(frame, pt1=(self.x_min, self.y_min - text_h - 2 * baseline),
                          pt2=(self.x_min + text_rect_w + 2 * baseline, self.y_min), color=color,
                          thickness=thickness)

        # draw the Rectangle
        cv2.rectangle(frame, (self.x_min, self.y_min), (self.x_max, self.y_max), color, thickness)

    def get_binary_mask(self, mask_shape):
        """
        Get uint8 binary mask of shape `mask_shape` with rectangle in foreground.
        Parameters
        ----------
        mask_shape : (tuple)
            Shape of the mask to return - following convention (h, w)
        Returns
        -------
        mask : np.array
            Binary uint8 mask of shape `mask_shape` with rectangle drawn as foreground.
        """
        if mask_shape[0] < self.y_max or mask_shape[1] < self.x_max:
            raise ValueError('Mask shape is smaller than Rectangle size')
        mask = np.zeros(shape=mask_shape, dtype=np.uint8)
        mask = cv2.rectangle(mask, self.tl_corner, self.br_corner, color=255, thickness=cv2.FILLED)
        return mask

    @property
    def tl_corner(self):
        """
        Coordinates of the top-left corner of rectangle (as int32).
        Returns
        -------
        tl_corner : int32 tuple
        """
        return tuple(map(np.int32, (self.x_min, self.y_min)))

    @property
    def br_corner(self):
        """
        Coordinates of the bottom-right corner of rectangle.

        Returns
        -------
        br_corner : int32 tuple
        """
        return tuple(map(np.int32, (self.x_max, self.y_max)))

    @property
    def coords(self):
        """
        Coordinates (x_min, y_min, x_max, y_max) which define the Rectangle.

        Returns
        -------
        coordinates : int32 tuple
        """
        return tuple(map(np.int32, (self.x_min, self.y_min, self.x_max, self.y_max)))

    @property
    def area(self):
        """
        Get the area of Rectangle

        Returns
        -------
        area : float32
        """
        return np.float32(self.x_side * self.y_side)
