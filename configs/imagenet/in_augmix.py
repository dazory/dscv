_base_ = [
    '../_base_/datasets/imagenet.py',
]

name = 'in_augmix'
num_inputs = 3

# data
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
]
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    train=dict(
        wrapper='AugMixDataset',
        pipeline=train_pipeline,
        preprocess=[dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)],
        augmentations=['autocontrast', 'equalize', 'posterize',
                       'rotate', 'solarize', 'shear_x', 'shear_y',
                       'translate_x', 'translate_y'],
        no_jsd=False,
    )
)

# model
model = dict(
    type='AugMixNet',
    model=dict(type='resnet50', pretrained=True),
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
    type='base_runner',
    epochs=90,
    work_dir=f'/ws/data2/dscv/{name}',
    evaluate=True,
    logger=dict(type='wandb_logger',
                init_kwargs=dict(
                    project='dscv',
                    entity='kaist-url-ai28',
                    name=name),
                use_wandb=True),
)
