_base_ = "../starlknet_b_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="resizemix"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)