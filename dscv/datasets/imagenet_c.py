from .imagenet import ImageNetDataset
from .builder import DATASETS


@DATASETS.register_module
class ImageNetCDataset(ImageNetDataset):
    def __init__(self,
                 data_root,
                 pipeline,
                 corruptions=None,
                 severities=None,
                 **kwargs,
                 ):
        self.data_root = data_root
        self.pipeline = pipeline

        assert isinstance(corruptions, list) and isinstance(severities, list)
        self.corruptions = corruptions
        self.severities = [1, 2, 3, 4, 5] if severities is None else severities

        data_c_init_root = f"{self.data_root}/{corruptions[0]}/{self.severities[0]}"
        super(ImageNetCDataset, self).__init__(data_c_init_root, pipeline)

    def set_mode(self, corruption, severity):
        data_c_root = f"{self.data_root}/{corruption}/{severity}"
        super(ImageNetCDataset, self).__init__(data_c_root, self.pipeline)
