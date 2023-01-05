import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def build_backbone(config):
    if config.type == 'vgg16':
        from torchvision.models import vgg16
        _backbone = vgg16(pretrained=True)
        backbone = nn.Sequential(*list(_backbone.features.children())[:-1])
    else:
        raise NotImplementedError

    return backbone


if __name__ == '__main__':
    import easydict
    from ..dataset.utils import build_dataset
    from dscv.utils.param_manager import ParamManager
    from ..utils import visualize_feature_summary

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

    backbone = build_backbone(config.model.backbone)

    dataset = build_dataset(config.dataset)
    image = dataset[0]['image']
    labels = dataset[0]['labels']
    bboxes = dataset[0]['bboxes']

    # Hook
    param_manager = ParamManager()
    # param_manager.check_param_structure(backbone)
    layer_name_list = [str(i) for i in range(30)]
    param_manager.register_forward_hook(backbone, layer_name_list, type='output')

    # Forward
    features = backbone(image.unsqueeze(0)).to(config.device)

    # Visualize
    for i in range(30):
        fig, ax = plt.subplots(1, 1)
        feature = param_manager.hook_results['output'][str(i)]
        visualize_feature_summary(feature.squeeze(0), fig, ax)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        fig.suptitle(f'backbone.{i}')
        plt.show(fig)
    param_manager.hook_results['output'] = {}
