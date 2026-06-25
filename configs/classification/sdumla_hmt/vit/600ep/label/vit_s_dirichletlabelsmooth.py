_base_ = "../vit_s_vanilla_sz224_bs32.py"

model = dict(
    head=dict(
        loss=dict(
            type='DirichletLabelSmoothLoss',
            alpha=0.1,
            label_smooth_val=0.1,
            num_classes=636,
            mode='soft',
            loss_weight=1.0
        ),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
