_base_ = [
    '../../../_base_/vera220/sz224_bs32_randerasing.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='BasicAugClassification',
    pretrained=None,
    aug_mode="vanilla",
    aug_args=dict(
        cutout=dict(),
        gridmask=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
                    cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        spnoise=dict(prob=0.1, noise_type='random'),
        randomblur=dict(),
        ricap=dict(choose_num=2,),
        yoco=dict(),
        softaugment=dict(t_crop=1.0, max_p_crop=1.0, pow_crop=2.0, bg_crop=1, sigma_crop=12,
                        iou=False, n_classes=220),
        vanilla=dict(),
    ),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7,),
        ),
    head=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=1280, num_classes=220),
)


# optimizer
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.05)

# use_fp16=True
# fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=5e-4)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)