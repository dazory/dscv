import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from model.backbone import build_backbone


class FasterRCNN(nn.Module):
    def __init__(self,
                 backbone,
                 rpn,
                 rcnn_proposal_target,
                 ):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.rcnn_proposal_target = rcnn_proposal_target



    def forward(self, data):
        '''
        Input:
            data   : (tensor) (B,
        '''
        images = data['image'].to(self.get_device())
        gt_bboxes = data['bbox'].to(self.get_device())
        gt_labels = data['label'].to(self.get_device())

        ##########################
        ### Feature extraction ###
        ##########################
        features = self.backbone(images)

        #######################
        ### Region Proposal ###
        #######################
        rois, rpn_cls_score, rpn_bbox_pred = self.rpn(features)

        self.roi_network(gt_bboxes, gt_labels, rois)


        roi_data = self.rcnn_proposal_target(rois, gt_bboxes)





        # RCNN
        rcnn_cls, rcnn_reg = self.roi()


        return






def build_faster_rcnn(device):
    import easydict

    config = easydict.EasyDict({
        'backbone': {
            'type': 'vgg16'
        },
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
        'roi': {
            'sampler': {
                'num_pos': 32,
                'num_total': 128,
                'pos_iou_thr': 0.5,
                'neg_iou_thr': 0.5,
            },
        },

    })

    backbone = build_backbone(config.backbone)
    classifier = nn.Sequential()

    rpn = RegionProposalNetwork()



    model = FasterRCNN()

    return model