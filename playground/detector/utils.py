import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


def visualize_feature(feature, num_vis=20, figsize_unit=(5, 3)):
    '''
    Input:
        feature : (tensor) (C, H, W).
    '''
    assert len(feature.shape) == 3

    cols = 5
    rows = math.ceil(num_vis / cols)
    h, w = figsize_unit[0] * cols, figsize_unit[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(h, w))

    c = 0
    for row in range(rows):
        for col in range(cols):
            if c >= len(feature):
                break
            axes[row, col].imshow(feature[c])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            c += 1


def visualize_feature_summary(feature,
                              fig=None, ax=None):
    '''
    Input:
        feature : (tensor) (C, H, W).
    '''
    assert len(feature.shape) == 3
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    feature = torch.mean(feature, 0)

    ax.imshow(feature.cpu().detach().numpy())

    return fig, ax


def get_color_array(cmap='rainbow', num_colors=8):
    if cmap == 'rainbow':
        cmap = cm.rainbow
    else:
        raise NotImplementedError
    COLOR_ARRAY = cmap(np.linspace(0, 1, num_colors))

    return COLOR_ARRAY


def visualize_bbox_xy(bbox,
                      fig=None, ax=None,
                      color=None, cmap='rainbow', num_colors=8, color_idx=None):
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    patch = Rectangle(xy=(x1,y1),
                      width=(x2-x1), height=(y2-y1),
                      linewidth=1,
                      edgecolor=color,
                      facecolor='none')
    ax.add_patch(patch)
    ax.plot()
    return fig, ax


def visualize_bboxes_xy(bboxes_xy,
                        fig=None, ax=None,
                        color=None, cmap='rainbow', num_colors=8, color_idx=None,
                        ):
    '''
    bboxes_xy   : (num_bboxes, 4) where 4 containing (x1,y1,x2,y2)
    '''
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    for bbox_xy in bboxes_xy:
        fig, ax = visualize_bbox_xy(bbox_xy, fig, ax, color, cmap, num_colors, color_idx)
    ax.plot()

    return fig, ax


def visualize_bbox_cxcy(bbox,
                        fig=None, ax=None,
                        color=None, cmap='rainbow', num_colors=8, color_idx=None):
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    patch = Rectangle(xy=(cx - w / 2., cy - h / 2.),
                      width=w, height=h,
                      linewidth=1,
                      edgecolor=color,
                      facecolor='none')
    ax.add_patch(patch)
    ax.plot()
    return fig, ax


def visualize_bboxes_cxcy(bboxes,
                          fig=None, ax=None,
                          color=None, cmap='rainbow', num_colors=8, color_idx=None):
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    for bbox_cxcy in bboxes:
        fig, ax = visualize_bbox_cxcy(bbox_cxcy, fig, ax, color, cmap, num_colors, color_idx)
    ax.plot()
    return fig, ax


def visualize_roi_process(bbox_pred, center_anchor, roi):
    num_colors = 3
    COLOR_ARRAY = cm.rainbow(np.linspace(0, 1, num_colors))

    fig, axes = plt.subplots(1, 4, figsize=(3 * 4, 3))

    # axes[0]
    visualize_bbox_cxcy(bbox_pred[0], ax=axes[0], color_idx=0, num_colors=3)
    axes[0].axes.set_title('bbox_pred from RPN')

    # axes[1]
    visualize_bbox_cxcy(bbox_pred[0], ax=axes[1], color_idx=0, num_colors=3)
    axes[1].plot([bbox_pred[0, 0]], [bbox_pred[0, 1]], '.', color=COLOR_ARRAY[0])
    visualize_bbox_cxcy(center_anchor[0], ax=axes[1], color_idx=1, num_colors=3)
    axes[1].plot([center_anchor[0, 0]], [center_anchor[0, 1]], '.', color=COLOR_ARRAY[1])
    axes[1].axes.set_title('Anchor')

    # axes[2]
    visualize_bbox_cxcy(bbox_pred[0], ax=axes[2], color_idx=0, num_colors=3)
    axes[2].plot([bbox_pred[0, 0]], [bbox_pred[0, 1]], '.', color=COLOR_ARRAY[0])
    visualize_bbox_cxcy(center_anchor[0], ax=axes[2], color_idx=1, num_colors=3)
    axes[2].plot([center_anchor[0, 0]], [center_anchor[0, 1]], '.', color=COLOR_ARRAY[1])
    visualize_bbox_cxcy(roi[0], ax=axes[2], color_idx=2, num_colors=3)
    axes[2].plot([roi[0, 0]], [roi[0, 1]], '.', color=COLOR_ARRAY[1])
    axes[2].axes.set_title('bbox_pred is decoded')

    # axes[3]
    visualize_bbox_cxcy(center_anchor[0], ax=axes[3], color_idx=1, num_colors=3)
    axes[3].plot([center_anchor[0, 0]], [center_anchor[0, 1]], '.', color=COLOR_ARRAY[1])
    visualize_bbox_cxcy(roi[0], ax=axes[3], color_idx=2, num_colors=3)
    axes[3].plot([roi[0, 0]], [roi[0, 1]], '.', color=COLOR_ARRAY[1])
    axes[3].axes.set_title('roi(decoded bbox_pred)')

#
# def visualize_anchor_base(anchor_base, fig=None, ax=None):
#     COLOR_ARRAY = cm.rainbow(np.linspace(0, 1, len(anchor_base)))
#
#     if (fig is None) and (ax is None):
#         fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     idx = 0
#     for anchor in anchor_base:
#         x1, y1, x2, y2 = anchor
#         patch = Rectangle(xy=(x1, y1),
#                           width=(x2-x1), height=(y2-y1),
#                           linewidth=2,
#                           edgecolor=COLOR_ARRAY[idx],
#                           facecolor='none')
#         ax.add_patch(patch)
#         idx += 1
#     ax.plot()
#     return fig, ax


def visualize_roi_tensor(center_anchor, bbox_pred, idx):
    '''
    Input:
        center_anchor   : (tensor) (num_anchors, 4) w/ cxcy
        bbox_pred       : (tensor) (num_preds, 4) w/ cxcy
        i               : (int) index of anchor and bbox_pred
    '''

    def decode(tcxcy, center_anchor):
        cxcy = tcxcy[:, :2] * center_anchor[:, 2:] + center_anchor[:, :2]
        wh = torch.exp(tcxcy[:, 2:]) * center_anchor[:, 2:]
        cxywh = torch.cat([cxcy, wh], dim=1)
        return cxywh

    def scale(tcxcy, center_anchor):
        cxcy = tcxcy[:, :2] * center_anchor[:, 2:]
        wh = torch.exp(tcxcy[:, 2:]) * center_anchor[:, 2:]
        cxywh = torch.cat([cxcy, wh], dim=1)
        return cxywh

    COLOR_ARRAY = cm.rainbow(np.linspace(0, 1, 4))

    fig, axes = plt.subplots(1, 4, figsize=(4 * 5, 5))

    # Visualize center_anchor
    visualize_bbox(center_anchor[idx], ax=axes[0], color_idx=0, title='center_anchor')
    axes[0].plot([center_anchor[idx, 0]], [center_anchor[idx, 1]], 'o', color=COLOR_ARRAY[0])
    visualize_bbox(center_anchor[idx], ax=axes[1], color_idx=0)
    axes[1].plot([center_anchor[idx, 0]], [center_anchor[idx, 1]], 'o', color=COLOR_ARRAY[0])
    visualize_bbox(center_anchor[idx], ax=axes[2], color_idx=0)
    axes[2].plot([center_anchor[idx, 0]], [center_anchor[idx, 1]], 'o', color=COLOR_ARRAY[0])
    visualize_bbox(center_anchor[idx], ax=axes[3], color_idx=0)

    # Visualize bbox_pred
    visualize_bbox(bbox_pred[idx], ax=axes[1], color_idx=1, title='bbox_pred')
    axes[1].plot([bbox_pred[idx, 0]], [bbox_pred[idx, 1]], 'o', color=COLOR_ARRAY[1])
    visualize_bbox(bbox_pred[idx], ax=axes[2], color_idx=1)
    axes[2].plot([bbox_pred[idx, 0]], [bbox_pred[idx, 1]], 'o', color=COLOR_ARRAY[1])
    visualize_bbox(bbox_pred[idx], ax=axes[3], color_idx=1)

    # Visualize bbox_pred scaled
    scaled_bbox_pred = scale(bbox_pred, center_anchor.to(bbox_pred.device))
    visualize_bbox(scaled_bbox_pred[idx], ax=axes[2], color_idx=2, title='scaled_bbox_pred')
    axes[2].plot([scaled_bbox_pred[idx, 0]], [scaled_bbox_pred[idx, 1]], 'o', color=COLOR_ARRAY[2])
    visualize_bbox(scaled_bbox_pred[idx], ax=axes[3], color_idx=2)
    axes[3].plot([scaled_bbox_pred[idx, 0]], [scaled_bbox_pred[idx, 1]], 'o', color=COLOR_ARRAY[2])

    # Visualize roi
    roi_tensor = decode(bbox_pred, center_anchor.to(bbox_pred.device))
    visualize_bbox(roi_tensor[idx], ax=axes[3], color_idx=3, title='roi')
    axes[3].plot([roi_tensor[idx, 0]], [roi_tensor[idx, 1]], 'o', color=COLOR_ARRAY[3])

    return fig, axes


