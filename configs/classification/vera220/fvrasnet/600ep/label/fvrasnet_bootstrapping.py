_base_ = "../fvrasnet_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        type='ClsHead',
        loss=dict(
            type='BootstrappingLoss',
            beta=0.1,
            num_classes=220,
            mode='soft',
            loss_weight=1.0
        ),
        with_avg_pool=True,
        multi_label=False,
        in_channels=256,
        num_classes=220
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
