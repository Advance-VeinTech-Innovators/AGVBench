_base_ = [
    '../../../_base_/vera220/sz224_bs32_vanilla.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='TeachAugment',
    pretrained=None,
    aug_module=dict(
        type='TeachAugModule',
        num_classes=220, scale=1, hidden=128, 
        n_dim=128, dropout_ratio=0.8, with_context=True,
        ),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7,),
        ),
    head_one=dict(
        type='ClsHead', 
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=1280, num_classes=220),
    head_aug=dict( 
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=1280, num_classes=220),
    head_weights=dict( head_one_q=1, head_aug_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',
        end_momentum=0.999999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1),
    dict(type='SAVEHook',
        iter_per_epoch=9,
        save_interval=9 * 100,
    )
]

# optimizer
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.05)

# use_fp16=True
# fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=5e-4)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)