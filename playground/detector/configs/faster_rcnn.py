import easydict
import torchvision

config = easydict.EasyDict({
        'dataset': {
            'root_dir': '/ws/data/cityscapes/leftImg8bit',
            'ann_file': '/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            'transforms': [torchvision.transforms.ToTensor()],
        },
        'model': {
            'backbone': {
                'type': 'vgg16',
            },
        },
        'device': 'cuda',
    })