import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorGenerator():
    def __init__(self,
                 base_size=16,
                 ratios=[0.5, 1.0, 2.0],
                 scales=[8, 16, 32]):
        super(AnchorGenerator, self).__init__()
        self.base_size = base_size
        self.ratios = ratios
        self.scales = scales

        self.anchor_base = self._generate_anchor_base()

    def _generate_anchor_base(self):
        px = self.base_size / 2.
        py = self.base_size / 2.

        anchor_base = torch.zeros((len(self.ratios) * len(self.scales), 4))
        idx = 0
        for ratio in self.ratios:
            for scale in self.scales:
                w = self.base_size * scale * np.sqrt(ratio)
                h = self.base_size * scale * np.sqrt(1 / ratio)

                # (x1, y1, x2, y2)
                anchor_base[idx, 0] = px - w / 2.
                anchor_base[idx, 1] = py - h / 2.
                anchor_base[idx, 2] = px + w / 2.
                anchor_base[idx, 3] = py + h / 2.
                idx += 1

        return anchor_base

    def generate_anchor(self, image_shape):
        '''
        Input:
            image_shape : (list) (H, W)
        Output:
            anchor  : (ndarray) (num_anchors, 4) where 4 containing [x1, y1, x2, y2].
        '''
        origin_height, origin_width = image_shape[0], image_shape[1]
        height, width = origin_height / self.base_size, \
                        origin_width / self.base_size
        feat_stride = self.base_size

        # shift_x : (width,) : [0, base_size, base_size*2, ..., base_size*(width-1)]
        # shift_y : (height,) : [0, base_size, base_size*2, ..., base_size*(height-1)]
        shift_x = torch.arange(0, width * feat_stride, feat_stride)
        shift_y = torch.arange(0, height * feat_stride, feat_stride)

        # shift_x = [[0, base_size*1, ..., base_size*(width-1)], ...]
        # shift_y = [[0,...,0], [base_size*1, ...], ..., [base_size*(height-1), ]]
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)  # (height, width)
        shift = torch.stack((shift_x.reshape(-1),
                             shift_y.reshape(-1),
                             shift_x.reshape(-1),
                             shift_y.reshape(-1)), dim=1)

        num_anchor_base = self.anchor_base.shape[0]
        num_shift = shift.shape[0]
        anchor = self.anchor_base.reshape((1, num_anchor_base, 4)) + shift.reshape((1, num_shift, 4)).permute(1, 0, 2)
        anchor = anchor.reshape((num_shift * num_anchor_base, 4))

        divisor = torch.tensor([origin_width, origin_height, origin_width, origin_height])
        anchor /= divisor

        return anchor


class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 anchor_generator,
                 image_shape,
                 in_channels=512,
                 feat_channels=512,
                 nms_pre=2000,
                 max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7),
                 min_bbox_size=0,
                 ):
        super(RegionProposalNetwork, self).__init__()
        cls_out_channels = 1 # [foreground] if 1 else [background, foreground]
        self.anchor_generator = anchor_generator

        self.image_shape = image_shape
        num_anchor_base = int(len(self.anchor_generator.anchor_base))

        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(feat_channels, num_anchor_base * cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, num_anchor_base * 4, 1)

        self.nms_pre = nms_pre

    def forward(self, features):
        out = self.rpn_conv(features)
        out = F.relu(out, inplace=True)
        rpn_cls_scores = self.rpn_cls(out)
        rpn_bbox_preds = self.rpn_reg(out)

        # Loss: using proposal_target_layer(pos, neg) in self.loss
        # if gt_labels is None:
        #     loss_inputs = outs + (gt_bboxes, img_metas)
        # else:
        #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # Proposal list: using nms in self.get_bboxes()
        # if proposal_cfg is None:
        #     return losses
        # else:
        #     proposal_list = self.get_bboxes(
        #         *outs, img_metas=img_metas, cfg=proposal_cfg)
        #     return losses, proposal_list
        output = self.rpn_proposal(rpn_cls_scores, rpn_bbox_preds)
        rois = output['rois']
        cls_scores = output['cls_scores']
        bbox_preds = output['bbox_preds']

        return rois, cls_scores, bbox_preds

    @staticmethod
    def _xy2cxcy(xy):
        '''
        Input:
            xy  : (tensor) (num_xy, 4)
        '''
        cxcy = (xy[:, 2:] + xy[:, :2]) / 2.
        wh = (xy[:, 2:] - xy[:, :2]) / 2.
        return torch.cat([cxcy, wh], dim=1)

    @staticmethod
    def _cxcy2xy(cxcy):
        '''
        Input:
            cxcy    : (tensor) (num_xy, 4)
        '''
        x1y1 = cxcy[:, :2] - cxcy[:, 2:] / 2.
        x2y2 = cxcy[:, :2] + cxcy[:, 2:] / 2.
        return torch.cat([x1y1, x2y2], dim=1)

    @staticmethod
    def _decode(tcxcy, center_anchor):
        '''
        Input:
            tcxcy           : (tensor) (B, num_anchors, 4)
            center_anchor   : (tensor) (num_anchors, 4)
        '''
        cxcy = tcxcy[:, :2] * center_anchor[:, 2:] + center_anchor[:, :2]
        wh = torch.exp(tcxcy[:, 2:]) * center_anchor[:, 2:]
        return torch.cat([cxcy, wh], dim=1)

    def rpn_proposal(self, rpn_cls_scores, rpn_bbox_preds):
        '''
        Input:
            rpn_cls_scores  : (tensor) shape=(B, num_classes*1, H', W')
            rpn_bbox_preds  : (tensor) shape=(B, num_classes*4, H', W')
        Output:
             rois           : (tensor) shape=(B, num_pre, 4)
             cls_scores     : (tensor) shape=(B, nms_pre)
             bbox_preds     : (tensor) shape=(B, nms_pre, 4)
        '''
        batch_size = rpn_cls_scores.shape[0]

        # Calculate class score(batch_size, num_anchors, 2)
        cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).contiguous()
        cls_scores = cls_scores.reshape(batch_size, -1)
        cls_scores = torch.sigmoid(cls_scores) # cls_score = torch.softmax(cls_score, dim=-1)

        # Get bounding box(batch_size, num_anchors, 4)
        bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        bbox_preds = bbox_preds.reshape(batch_size, -1, 4)

        # Get anchors
        anchors = self.anchor_generator.generate_anchor(self.image_shape).to(cls_scores.device)
        center_anchors = self._xy2cxcy(anchors)

        # Decode as roi tensor
        roi_list = []
        cls_score_list = []
        bbox_pred_list = []
        center_anchor_list = []
        anchor_list = []
        for b in range(batch_size):
            cls_score = cls_scores[b]
            bbox_pred = bbox_preds[b]

            ranked_scores, rank_inds = cls_score.sort(descending=True)
            topk_inds = rank_inds[:self.nms_pre]

            cls_score = ranked_scores[:self.nms_pre]
            bbox_pred = bbox_pred[topk_inds, :]
            center_anchor = center_anchors[topk_inds, :]
            anchor = anchors[topk_inds, :]

            # H, W = self.image_shape
            # scale = torch.tensor([W, H, W, H]).to(bbox_pred.get_device())
            roi = self._decode(bbox_pred, center_anchor)#  * scale

            cls_score_list.append(cls_score)
            bbox_pred_list.append(bbox_pred)
            roi_list.append(roi)
            center_anchor_list.append(center_anchor)
            anchor_list.append(anchor)
        cls_scores = torch.stack(cls_score_list, dim=0)
        bbox_preds = torch.stack(bbox_pred_list, dim=0)
        rois = torch.stack(roi_list, dim=0)
        center_anchors = torch.stack(center_anchor_list, dim=0)
        anchors = torch.stack(anchor_list, dim=0)

        output = {
            'rois': rois,
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'anchors': anchors,
        }
        return output # rois, cls_scores, bbox_preds


def build_anchor_generator(config):
    anchor_generator = AnchorGenerator(
        base_size=config.base_size,
        ratios=config.ratios,
        scales=config.scales,
    )
    return anchor_generator


def build_rpn(config):
    anchor_generator = build_anchor_generator(config.anchor)

    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        image_shape=config.image_shape,
        in_channels=config.in_channels,
        feat_channels=config.feat_channels,
        nms_pre=config.nms_pre,
    )

    return rpn


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ..configs.faster_rcnn import *
    from ..utils import visualize_bboxes_xy

    config['model'].update({
        'rpn': {
            'image_shape': [1024, 2048],
            'in_channels': 512,
            'feat_channels': 512,
            'anchor': {
                'base_size': 16,
                'ratios': [0.5, 1.0, 2.0],
                'scales': [8, 16, 32],
            },
            'nms_pre': 2000,
        },
    })

    ########################
    ### Anchor Generator ###
    ########################
    anchor_generator = build_anchor_generator(config.model.rpn.anchor).to(config.device)
    anchor = anchor_generator.generate_anchor(config.model.rpn.image_shape)

    ### Visualize: anchor base ###
    anchor_base = anchor_generator.anchor_base
    print(f"Generate anchor base... ")
    print(f"  Output: ")
    print(f"  > anchor_base : {anchor_base.shape}")
    print(anchor_base)

    fig, ax = visualize_bboxes_xy(anchor_base)
    ax.plot([anchor_generator.base_size / 2], [anchor_generator.base_size / 2], '.', color='black')
    ax.annotate(f'({int(anchor_generator.base_size / 2)}, {int(anchor_generator.base_size / 2)})',
                (anchor_generator.base_size / 2, anchor_generator.base_size / 2))
    fig.suptitle('Anchor base generated by AnchorGenerator')
    plt.show(fig)

    ### Visualize: anchor ###
    fig, ax = plt.subplots(1, 1)
    num_vis = 5
    for i in range(num_vis):
        fig, ax = visualize_bboxes_xy(anchor[9 * i: 9 * (i + 1), :],
                                      fig=fig, ax=ax,
                                      color_idx=i, num_colors=num_vis)
    fig.suptitle('Anchor generated by AnchorGenerator')
    plt.show(fig)



