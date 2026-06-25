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
        type='SwinTransformer',
        arch="tiny",
        drop_path_rate=0.1,
        out_indices=(3,)
    ),
    head_one=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=768, num_classes=636),
    head_aug=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=768, num_classes=636),
    head_weights=dict(head_one_q=1, head_aug_k=1),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer=['LayerNorm', 'BatchNorm'], val=1., bias=0.)
    ],
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
    lr=1e-4,
    weight_decay=0.001, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.),
    })

# interval for accumulate gradient
update_interval = 1  # total: 1 x bs100 x 1 accumulates = bs100

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.0,
    warmup='linear',
    warmup_iters=150, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
