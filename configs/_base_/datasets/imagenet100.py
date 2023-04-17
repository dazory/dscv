dataset_type = 'ImageNetDataset'
data_root = '/ws/data/imagenet100'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    batch_size=256, num_workers=8,
    train=dict(
        type=dataset_type,
        data_root=f"{data_root}/train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=f"{data_root}/val",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=f"{data_root}/val",
        pipeline=test_pipeline,
    )
)
