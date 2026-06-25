_base_ = "../swin_t_mixups_sz224_bs32.py"

model = dict(
    mix_mode="guidedmix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
