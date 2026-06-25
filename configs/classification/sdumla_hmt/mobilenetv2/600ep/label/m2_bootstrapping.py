_base_ = "../m2_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        loss=dict(
            type='BootstrappingLoss',
            beta=0.1,
            num_classes=636,
            mode='soft',
            loss_weight=1.0
        ),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
