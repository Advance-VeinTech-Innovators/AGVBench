_base_ = "../starlknet_s_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        loss=dict(
            type='ConfidencePenaltyLoss',
            lambda_cp=0.1,
            num_classes=1100,
            mode='mean',
            loss_weight=1.0
        ),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
