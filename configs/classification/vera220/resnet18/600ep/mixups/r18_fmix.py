_base_ = "../r18_mixups_sz224_bs32.py"

model = dict(
    alpha=0.2,
    mix_mode="fmix"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)