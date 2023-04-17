from torchvision import transforms


type_to_op = {
    'RandomResizedCrop': lambda size: transforms.RandomResizedCrop(size),
    'RandomHorizontalFlip': lambda : transforms.RandomHorizontalFlip(),
    'Resize': lambda size: transforms.Resize(size),
    'CenterCrop': lambda size: transforms.CenterCrop(size),
    'ToTensor': lambda : transforms.ToTensor(),
    'Normalize': lambda mean, std: transforms.Normalize(mean, std),
}


def build_pipeline(pipeline=None):
    if pipeline is None:
        pipeline = []
    assert isinstance(pipeline, list)
    _pipeline = []
    for t in pipeline:
        _type = t.pop('type')
        _pipeline.append(type_to_op[_type](**t))
    return transforms.Compose(_pipeline)
