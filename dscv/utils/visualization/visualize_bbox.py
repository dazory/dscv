import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

from dscv.utils.color import COLOR_ARRAY_LIST


'''
Visualize bounding boxes
    Return: fig, ax
'''


def visualize_bbox_xy(bbox, fig=None, ax=None,
                      linewidth=1, edgecolor=None, **kwargs):
    '''
    bbox        : (4,) where 4 containing (x1,y1,x2,y2)
    fig, ax     : matplotlib figure and axis
    kwargs      : keyword arguments for matplotlib.patches.Rectangle
    '''

    # Generate figure and axis if not provided
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    patch = Rectangle(xy=(x1, y1), width=(x2-x1), height=(y2-y1),
                      linewidth=linewidth, edgecolor=edgecolor, facecolor='none', **kwargs)
    ax.add_patch(patch)
    ax.plot()
    return fig, ax


def visualize_bboxes_xy(bboxes_xy, fig=None, ax=None,
                        color=None, num_colors=-1, color_idx=None, **kwargs):
    '''
    bboxes_xy   : (num_bboxes, 4) where 4 containing (x1,y1,x2,y2)
    fig, ax     : matplotlib figure and axis
    kwargs      : keyword arguments for matplotlib.patches.Rectangle
                    E.g., linewidth=3
    '''

    # Generate figure and axis if not provided
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    # Generate color if not provided
    if color is None:
        color = COLOR_ARRAY_LIST['default'] if num_colors < 0 \
            else cm.rainbow(np.linspace(0, 1, num_colors))[color_idx]

    # Visualize bboxes
    for bbox_xy in bboxes_xy:
        fig, ax = visualize_bbox_xy(bbox_xy, fig, ax, edgecolor=color, **kwargs)
    ax.plot()

    return fig, ax


def visualize_bbox_cxcy(bbox, fig=None, ax=None,
                        linewidth=1, edgecolor=None, **kwargs):
    '''
    bbox        : (4,) where 4 containing (cx, cy, w, h)
    fig, ax     : matplotlib figure and axis
    kwargs      : keyword arguments for matplotlib.patches.Rectangle
    '''

    # Generate figure and axis if not provided
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    patch = Rectangle(xy=(cx - w / 2., cy - h / 2.), width=w, height=h,
                      linewidth=linewidth, edgecolor=edgecolor, facecolor='none', **kwargs)
    ax.add_patch(patch)
    ax.plot()
    return fig, ax


def visualize_bboxes_cxcy(bboxes_cxcy, fig=None, ax=None,
                          color=None, num_colors=-1, color_idx=None, **kwargs):
    '''
    bboxes_cxcy : (num_bboxes, 4) where 4 containing (cx,cy,w,h)
    fig, ax     : matplotlib figure and axis
    kwargs      : keyword arguments for matplotlib.patches.Rectangle
                    E.g., linewidth=3
    '''

    # Generate figure and axis if not provided
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    # Generate color if not provided
    if color is None:
        color = COLOR_ARRAY_LIST['default'] if num_colors < 0 \
            else cm.rainbow(np.linspace(0, 1, num_colors))[color_idx]

    # Visualize bboxes
    for bbox_cxcy in bboxes_cxcy:
        fig, ax = visualize_bbox_cxcy(bbox_cxcy, fig, ax, edgecolor=color, **kwargs)
    ax.plot()

    return fig, ax
