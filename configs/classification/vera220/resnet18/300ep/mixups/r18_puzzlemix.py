_base_ = "../r18_mixups_sz224_bs32.py"

model = dict(
    alpha=2.0,
    mix_mode="puzzlemix"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)