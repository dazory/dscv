import torchvision

from .builder import DATASETS


@DATASETS.register_module
class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self,
                 data_root,
                 pipeline,
                 **kwargs,
                 ):
        super(ImageNetDataset, self).__init__(data_root, pipeline)


