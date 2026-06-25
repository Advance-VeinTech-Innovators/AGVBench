_base_ = "../r18_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="guidedmix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)