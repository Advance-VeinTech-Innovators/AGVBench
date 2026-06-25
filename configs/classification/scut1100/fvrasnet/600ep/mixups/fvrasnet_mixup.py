_base_ = "../fvrasnet_mixups_sz224_bs32.py"

model = dict(
    aug_mode="mixup"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)