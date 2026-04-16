_base_ = [
    '../../_base_/vera220/sz224_bs32_vanilla.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    alpha=[1, 0.8],
    mix_mode=["mixup", "cutmix"],
    mix_args=dict(
        augmix=dict(mixture_depth=-1, mixture_width=3, severity=1),
        fmix=dict(decay_power=3, size=(224,224), max_soft=0., reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
            cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(transport=True, t_batch_size=32, t_size=-1,
            mp=None, block_num=4,
            beta=1.2, gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
    ),
    backbone=dict(
        type='MetaFormer',
        arch='poolformerv2_s36',
        drop_path_rate=0.3,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=220, mode='original', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=220),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer=['LayerNorm', 'GroupNorm'], val=1., bias=0.)
    ],
)

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=1e-2, eps=1e-6, betas=(0.9, 0.999))

optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
