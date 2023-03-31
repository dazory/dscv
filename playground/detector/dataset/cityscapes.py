import os
import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image

PIXEL2INCH = 0.0104166667


class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms, split='train'):
        super(CocoDataset, self).__init__()
        '''
        Input:
            root_dir    : (str) E.g., '/ws/data/cityscapes/leftImg8bit'
            ann_file    : (str) E.g., '/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'
            transforms  : (list) contains transform function. E.g., [torchvision.transforms.ToTensor()]
        '''
        self.root_dir = root_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.split = split

        self.CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        self.COLOR_ARRAY = cm.rainbow(np.linspace(0, 1, len(self.CLASSES)))

        self.coco = COCO(os.path.join(self.ann_file))
        self.img_ids = self.coco.getImgIds()

        self.data_infos = []
        self.ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs(ids=i)[0]
            info['filename'] = info['file_name']
            self.data_infos.append(info)

            ann_id = self.coco.getAnnIds(imgIds=i)
            self.ann_ids.append(ann_id)

        cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.img_ids)

    ''' Get Item '''
    def __getitem__(self, idx):
        data = dict()

        # Image
        data['image'] = self.load_image(idx)

        # Annotations: Bounding box and labels
        annotations = self.load_annotations(idx)
        bboxes = torch.FloatTensor(annotations[:, :4])
        labels = torch.FloatTensor(annotations[:, 4])
        data['bboxes'] = bboxes
        data['labels'] = labels

        return data

    def load_image(self, idx):
        '''
        Output:
            image: (PIL)
        '''
        img_info = self.data_infos[idx]
        file_path = os.path.join(self.root_dir, img_info['filename'])

        image = Image.open(file_path).convert('RGB')

        for transform in self.transforms:
            image = transform(image)

        return image

    def load_annotations(self, idx):
        ann_id = self.ann_ids[idx]
        anns = self.coco.loadAnns(ann_id)

        annotations = np.zeros((0, 5))
        for i, anno_dict in enumerate(anns):
            if anno_dict['bbox'][2] < 1 or anno_dict['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            # (x1, y1, w, h) -> (cx, cy, w, h)
            x1, y1, w, h = anno_dict['bbox']
            annotation[0, 0] = x1 + w / 2.
            annotation[0, 1] = y1 + h / 2.
            annotation[0, 2] = w
            annotation[0, 3] = h
            annotation[0, 4] = anno_dict['category_id']
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    ''' Helper '''
    @staticmethod
    def xywh2xyxy(x, y=None, w=None, h=None):
        if y is None:
            x, y, w, h = x[0], x[1], x[2], x[3]
        x1 = x - w
        x2 = x + w
        y1 = y - h
        y2 = y + h
        return (x1, x2, y1, y2)

    @staticmethod
    def xyxy2xywh(x1, y1=None, x2=None, y2=None):
        if x2 is None:
            x1, y1, x2, y2 = x1[0], x1[1], x1[2], x1[3]
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        w = (x2 - x1) / 2.0
        h = (y2 - y1) / 2.0
        return (x, y, w, h)


class CityscapesDataset(CocoDataset):

    def __init__(self, root_dir, ann_file, transforms, split='train'):
        super(CityscapesDataset, self).__init__(root_dir, ann_file, transforms, split)

        self.CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                   'bicycle')
        self.num_classes = len(self.CLASSES)
        self.COLOR_ARRAY = cm.rainbow(np.linspace(0, 1, len(self.CLASSES)))

    def load_image(self, idx):
        '''
        Output:
            image: (PIL)
        '''
        img_info = self.data_infos[idx]
        file_path = os.path.join(self.root_dir, self.split,
                                 img_info['filename'])

        image = Image.open(file_path).convert('RGB')

        for transform in self.transforms:
            image = transform(image)

        return image

    def __getitem__(self, idx):
        data = dict()

        # Image
        image = self.load_image(idx)
        data['image'] = image
        data['img_shape'] = image.shape

        # Annotations: Bounding boxes and labels
        annotations = self.load_annotations(idx)
        bboxes = torch.FloatTensor(annotations[:, :4])
        labels = torch.FloatTensor([self.cat2label[label] for label in annotations[:,4]])
        data['bboxes'] = bboxes
        data['labels'] = labels

        return data

