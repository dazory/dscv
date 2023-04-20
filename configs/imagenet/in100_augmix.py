_base_ = [
    '../_base_/schedule.py',
    '../_base_/datasets/imagenet100.py',
]

name = 'in100_augmix'
num_inputs = 3

# data
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
]
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # batch_size=128,
    train=dict(
        wrapper='AugMixDataset',
        pipeline=train_pipeline,
        preprocess=[dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)],
        augmentations=['autocontrast', 'equalize', 'posterize',
                       'rotate', 'solarize', 'shear_x', 'shear_y',
                       'translate_x', 'translate_y'],
        no_jsd=False,
    ),
    val_c=dict(
        type='ImageNetCDataset',
        data_root=f"/ws/data/imagenet100-c",
        pipeline=test_pipeline,
        corruptions=['gaussian_noise', 'shot_noise', 'impulse_noise',
                     'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog',
                     'brightness', 'contrast', 'elastic_transform',
                     'pixelate', 'jpeg_compression'],
        severities=[1, 2, 3, 4, 5]
    )
)

# model
model = dict(
    type='AugMixNet',
    model=dict(type='resnet50', pretrained=True),
    num_inputs=num_inputs,
    loss=[
        dict(type='CrossEntropyLoss',
             name='orig_loss', weight=1.0
             ),
        dict(type='JSDivLoss',
             name='dg_loss', weight=12.0,
             num_inputs=num_inputs,
             ),
    ],
)

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3])

runner = dict(
    type='RunnerC',
    epochs=90,
    work_dir=f'/ws/data2/dscv/{name}',
    modes=['train', 'val', 'val_c'],
    logger=dict(type='wandb_logger',
                init_kwargs=dict(
                    project='dscv',
                    entity='kaist-url-ai28',
                    group=name,
                    name=name),
                use_wandb=True),
)
