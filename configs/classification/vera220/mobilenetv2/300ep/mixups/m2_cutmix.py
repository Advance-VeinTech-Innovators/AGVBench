_base_ = "../m2_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="mixup"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)