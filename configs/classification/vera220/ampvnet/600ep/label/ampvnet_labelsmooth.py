_base_ = "../ampvnet_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        type='ClsHead',
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=220,
            mode='original',
            loss_weight=1.0
        ),
        with_avg_pool=True,
        multi_label=False,
        in_channels=512,
        num_classes=220
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
