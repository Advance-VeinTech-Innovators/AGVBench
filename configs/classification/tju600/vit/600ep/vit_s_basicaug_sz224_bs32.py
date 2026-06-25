_base_ = [
    '../../../_base_/tju600/sz224_bs32_vanilla.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='BasicAugClassification',
    pretrained=None,
    aug_mode="cutout",
    aug_args=dict(
        cutout=dict(),
        gridmask=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
                      cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        spnoise=dict(prob=0.1, noise_type='random'),
        randomblur=dict(),
        randnquant=dict(region_num=4, collapse_to_val='inside_random', spacing='random'),
        yoco=dict(),
        softaugment=dict(t_crop=1.0, max_p_crop=1.0, pow_crop=2.0, bg_crop=1, sigma_crop=12,
                         iou=False, n_classes=600),
        vanilla=dict(),
        ricap=dict(choose_num=2, ),
    ),
    backbone=dict(
        type='VisionTransformer',
        arch="small",
        drop_path=0.1,
    ),
    head=dict(
        type='VisionTransformerClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        multi_label=False, in_channels=768, num_classes=600),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer=['LayerNorm', 'BatchNorm'], val=1., bias=0.)
    ],
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.001, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
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
