from .cityscapes import CityscapesDataset


def build_dataset(config):
    dataset = CityscapesDataset(config.root_dir,
                                config.ann_file,
                                config.transforms, split='train')
    return dataset


if __name__ == '__main__':
    import easydict
    from ..utils import visualize_bbox_cxcy
    import torchvision
    import matplotlib.pyplot as plt

    config = easydict.EasyDict({
        'dataset': {
            'root_dir': '/ws/data/cityscapes/leftImg8bit',
            'ann_file': '/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            'transforms': [torchvision.transforms.ToTensor()],
        },
        'device': 'cuda',
    })

    dataset = build_dataset(config.dataset)

    image = dataset[0]['image']
    labels = dataset[0]['labels']
    bboxes = dataset[0]['bboxes']

    fig, ax = plt.subplots(1,1)
    for i in range(len(bboxes)):
        fig, ax = visualize_bbox_cxcy(bboxes,
                                      fig, ax,
                                      num_colors=dataset.num_classes,
                                      color_idx=int(labels[i]))
    plt.show(fig)