_base_ = "../swin_t_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1100,
            mode='original',
            loss_weight=1.0
        ),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
