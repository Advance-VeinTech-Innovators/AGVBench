_base_ = [
    '../../../_base_/sdumla_hmt/sz224_bs32_vanilla.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='MAdAugmentation',
    pretrained=None,
    aug_module=dict(
        type='MAdAugBlock',
        num_classes=636, noise_std=0.1, in_channels=3,
        subpolicies=13, threshold=0.2,
    ),
    backbone=dict(
        type='AMPVNet',
        out_indices=(3,),
    ),
    head_one=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=636),
    head_aug=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=636),
    head_weights=dict(head_one_q=1, head_aug_k=0.5),
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
         iter_per_epoch=9,
         save_interval=9 * 100,
         )
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=1e-2,
    eps=1e-6,
    betas=(0.9, 0.999),
    paramwise_options={
        # 'aug_module.p_aug' 是 PolicyNetwork 在模型中的路径
        'aug_module.p_aug': dict(lr=0.00001),
    }
)

# use_fp16=True
# fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2), update_interval=1)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
