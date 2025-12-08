_base_ = "../starlknet_b_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="cutout"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)