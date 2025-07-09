_base_ = "../m2_mixups_sz224_bs32.py"

model = dict(
    alpha=0.2,
    mix_mode="gridmix"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)