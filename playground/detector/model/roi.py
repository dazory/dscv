import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool


class BBoxRoIExtractor(nn.Module):
    def __init__(self,
                 out_size=(7, 7)
                 ):
        super(BBoxRoIExtractor, self).__init__()
        self.roi_pool = RoIPool(output_size=out_size, spatial_scale=1.)

    @staticmethod
    def _cxcy2xy(cxcy):
        '''
        Input:
            cxcy    : (tensor) (num_xy, 4)
        '''
        x1y1 = cxcy[:, :2] - cxcy[:, 2:] / 2.
        x2y2 = cxcy[:, :2] + cxcy[:, 2:] / 2.
        return torch.cat([x1y1, x2y2], dim=1)

    def forward(self, features, rois):
        '''
        Input:
            features    : (tensor) (B, num_feats, H', W')
            rois        : (tensor) (B, num_samples, 4)
        Output:
            bbox_feat_list  : (list) contains bbox_feat(tensor)
                bbox_feat: (tensor) shape=(num_samples, num_feats, out_size[0], out_size[1])
        '''
        feat_height, feat_width = features.shape[-2], features.shape[-1]
        scale = torch.tensor([feat_width, feat_height,
                              feat_width, feat_height]).to(rois.get_device())
        scaled_rois = rois * scale

        scaled_roi_list = []
        for b in range(len(rois)):
            scaled_roi = scaled_rois[b]
            scaled_roi = self._cxcy2xy(scaled_roi)
            scaled_roi_list.append(scaled_roi)

        bbox_feats = self.roi_pool(features, scaled_roi_list)
        bbox_feat_list = list(torch.chunk(bbox_feats, chunks=len(rois), dim=0))

        return bbox_feat_list


class RoIBboxHead(nn.Module):
    def __init__(self,
                 in_features,
                 num_classes,
                 out_features=4096,
                 ):
        super(RoIBboxHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.ReLU(inplace=True),
        )
        self.fc_cls = nn.Linear(out_features, num_classes)
        self.fc_reg = nn.Linear(out_features, num_classes * 4)

    def forward(self, bbox_feats):
        '''
        Input:
            bbox_feats   : (tensor) shape=(B, num_rois, num_feats, out_size[0], out_size[1])
        '''
        batch_size, num_rois = bbox_feats.shape[0], bbox_feats.shape[1]
        bbox_feats = bbox_feats.reshape(batch_size, num_rois, -1)

        cls_feats = self.classifier(bbox_feats)

        cls_scores = self.fc_cls(cls_feats)
        bbox_preds = self.fc_reg(cls_feats)

        return cls_scores, bbox_preds


class RoINetwork(nn.Module):
    def __init__(self,
                 bbox_roi_extractor,
                 bbox_head,
                 num_pos=32,
                 num_total=128,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5,
                 image_shape=[1024, 2048],
                 ):
        super(RoINetwork, self).__init__()
        self.num_pos = num_pos
        self.num_total = num_total
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

        self.image_shape = image_shape

        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head

    @staticmethod
    def _calculate_overlaps(bbox1, bbox2):
        '''
        Input:
            bbox1   : (tensor) (num_bboxes1, 4) w/ xy form
            bbox2   : (tensor) (num_bboxes2, 4) w/ xy form
        Output:
            overlaps    : (tensor) (num_bboxes1, num_bboxes2) w/ xy form
        '''
        # lower_bounds = {(x1, y1): x1=max(bbox1[:,0], bbox2[:,0]), y1=max(bbox1[:,1], bbox2[:,1])}
        # upper_bounds = {(x2, y2): x2=min(bbox1[:,2], bbox2[:,2]), y2=min(bbox1[:,3], bbox2[:,3])}
        lower_bounds = torch.max(bbox1[:, :2].unsqueeze(1),
                                 bbox2[:, :2].unsqueeze(0))
        upper_bounds = torch.min(bbox1[:, 2:].unsqueeze(1),
                                 bbox2[:, 2:].unsqueeze(0))

        overlaps = torch.clamp(upper_bounds - lower_bounds, min=0)
        overlaps = overlaps[:, :, 0] * overlaps[:, :, 1]

        return overlaps

    @staticmethod
    def _calculate_areas(bboxes):
        '''
        Input:
            bboxes  : (tensor) (num_bboxes, 4) w/ xy form
        Output:
            areas   : (tensor) (num_bboxes)
        '''
        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]
        areas = torch.clamp(width * height, min=0)
        return areas

    def calculate_ious(self, bbox1, bbox2):
        '''
        Input:
            bbox1   : (tensor) (num_bboxes1, 4) w/ xy form
            bbox2   : (tensor) (num_bboxes2, 4) w/ xy form
        Output:
            ious    : (tensor) (num_bboxes1, num_bboxes2)
        '''
        overlaps = self._calculate_overlaps(bbox1, bbox2)

        area_bbox1 = self._calculate_areas(bbox1).unsqueeze(1)
        area_bbox2 = self._calculate_areas(bbox2).unsqueeze(0)

        union = area_bbox1 + area_bbox2 - overlaps

        return overlaps / union

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

    def encode(self, gt_cxcy, anc_cxcy):
        tg_cxy = (gt_cxcy[..., :2] - anc_cxcy[..., :2]) / anc_cxcy[..., 2:]
        tg_wh = torch.log(gt_cxcy[..., 2:] / anc_cxcy[..., 2:])
        tg_cxywh = torch.cat([tg_cxy, tg_wh], dim=1)
        return tg_cxywh

    def sample(self, gt_bboxes, gt_labels, rois):
        '''
        Input:
            gt_bboxes   : (tensor) (B, num_gts, 4)
            gt_labels   : (tensor) (B, num_gts)
            rois        : (tensor) (B, num_rois(=nms_pre), 4)
        Output:
            labels      : (tensor) (B, num_total)
            bboxes      : (tensor) (B, num_total, 4)
            rois        : (tensor) (B, num_total, 4)
        '''

        batch_size = len(gt_labels)
        label_list = []
        bbox_list, neg_bbox_list, pos_bbox_list = [], [], []
        pos_ind_list, neg_ind_list = [], []
        roi_list = []
        for b in range(batch_size):
            gt_bbox = gt_bboxes[b]
            gt_label = gt_labels[b]
            roi = rois[b]

            # iou : (num_rois+num_gts, num_gts)
            H, W = self.image_shape
            scale = torch.tensor([W, H, W, H]).to(roi.get_device())
            roi = torch.cat([roi, gt_bbox/scale], dim=0)
            iou = self.calculate_ious(roi, gt_bbox)
            iou_max, iou_argmax = iou.max(dim=1)
            num_ious = len(iou)

            # +1 makes background as label of zero
            label = gt_label[iou_argmax] + 1

            num_pos = int(min((iou_max >= self.pos_iou_thr).sum(), self.num_pos))
            pos_ids = torch.arange(num_ious)[iou_max >= self.pos_iou_thr]
            rand_perm = torch.randperm(len(pos_ids))
            pos_ids = pos_ids[rand_perm[:num_pos]]

            num_neg = self.num_total - num_pos
            neg_ids = torch.arange(num_ious)[(iou_max < self.neg_iou_thr) & (iou_max >= 0.0)]
            rand_perm = torch.randperm(len(neg_ids))
            neg_ids = neg_ids[rand_perm[:num_neg]]

            assert (num_pos + num_neg) == self.num_total

            sampled_ids = torch.cat([pos_ids, neg_ids], dim=-1)
            label = label[sampled_ids]
            label[num_pos:] = 0  # seg negative indices background label
            label = label.type(torch.long)

            target_bbox = gt_bbox[iou_argmax][sampled_ids]
            sampled_roi = roi[sampled_ids, :]
            encoded_roi = self.encode(self._xy2cxcy(target_bbox),
                                      self._xy2cxcy(sampled_roi))

            # Normalize bbox
            mean = torch.FloatTensor([0., 0., 0., 0.]).to(encoded_roi.get_device())
            std = torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(encoded_roi.get_device())
            encoded_roi = (encoded_roi - mean) / std

            label_list.append(label)
            bbox_list.append(encoded_roi)
            neg_bbox_list.append(encoded_roi[:num_pos])
            pos_bbox_list.append(encoded_roi[num_pos:])
            pos_ind_list.append(pos_ids)
            neg_ind_list.append(neg_ids)
            roi_list.append(sampled_roi)

        output = {
            'labels': torch.stack(label_list, dim=0),
            'bboxes': torch.stack(bbox_list, dim=0),
            'neg_bboxes': torch.stack(neg_bbox_list, dim=0),
            'pos_bboxes': torch.stack(pos_bbox_list, dim=0),
            'pos_inds': torch.stack(pos_ind_list, dim=0),
            'neg_inds': torch.stack(neg_ind_list, dim=0),
            'rois': torch.stack(roi_list, dim=0),
        }

        return output

    def forward(self,
                gt_bboxes,
                gt_labels,
                rois,
                features):
        '''
        Input:
            gt_bboxes   : (tensor) (B, num_gts, 4)
            gt_labels   : (tensor) (B, num_gts)
            rois        : (tensor) (B, num_pre, 4)
            features    : (tensor) (B, num_feats, H', W')
        Output:
            cls_scores  : (tensor) (B, num_total, num_classes)
            bbox_preds  : (tensor) (B, num_total, num_classes*4)
        '''

        sampled_result = self.sample(gt_bboxes, gt_labels, rois)
        target_labels = sampled_result['labels']    # (B, num_samples), where num_samples=num_pos+num_neg
        target_bboxes = sampled_result['bboxes']    # (B, num_samples, 4)
        sampled_rois = sampled_result['rois']       # (B, num_samples, 4)

        # bbox_feat_list  : (list) contains bbox_feat(tensor),
        #                   where bbox_feat: (tensor) shape=(num_samples, num_feats, out_size[0], out_size[1])
        bbox_feat_list = self.bbox_roi_extractor(features, sampled_rois)
        bbox_feats = torch.stack(bbox_feat_list, dim=0)

        cls_scores, bbox_preds = self.bbox_head(bbox_feats)

        # Reshape: cls_scores=(B, num_total), bbox_preds=(B, num_total, 4)
        ''' Test mode...
        cls_score_list, bbox_pred_list = [], []
        for b in range(len(cls_scores)):
            cls_score = cls_scores[b]
            max_inds = torch.max(cls_score, dim=-1).indices
            cls_score = cls_score[torch.arange(len(cls_score)), max_inds]
            bbox_pred = bbox_preds[b].reshape(len(cls_score), -1, 4)
            bbox_pred = bbox_pred[torch.arange(len(cls_score)), max_inds, :]

            cls_score_list.append(cls_score)
            bbox_pred_list.append(bbox_pred)
        cls_scores = torch.stack(cls_score_list, dim=0)
        bbox_preds = torch.stack(bbox_pred_list, dim=0)
        '''
        return cls_scores, bbox_preds


def build_roi(config):

    bbox_roi_extractor = BBoxRoIExtractor(out_size=config.out_size)
    bbox_head = RoIBboxHead(
        in_features=config.bbox_head.num_channels * config.out_size[0] * config.out_size[1],
        num_classes=config.bbox_head.num_classes,
    )

    roi = RoINetwork(
        num_pos=config.sampler.num_pos,
        num_total=config.sampler.num_total,
        pos_iou_thr=config.sampler.pos_iou_thr,
        neg_iou_thr=config.sampler.neg_iou_thr,
        bbox_roi_extractor=bbox_roi_extractor,
        bbox_head=bbox_head,
    )
    return roi
