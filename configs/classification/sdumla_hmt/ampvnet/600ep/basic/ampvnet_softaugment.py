_base_ = "../ampvnet_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="softaugment",
    aug_args=dict(
        softaugment=dict(t_crop=1.0, max_p_crop=1.0, pow_crop=2.0, bg_crop=1, sigma_crop=12,
                         iou=False, n_classes=636),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
