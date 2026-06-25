_base_ = "../m2_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        type='ClsHead',
        loss=dict(
            type='DirichletLabelSmoothLoss',
            alpha=0.1,
            label_smooth_val=0.1,
            num_classes=492,
            mode='soft',
            loss_weight=1.0
        ),
        with_avg_pool=True,
        multi_label=False,
        in_channels=1280,
        num_classes=492
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
