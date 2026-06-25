_base_ = "../swin_t_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="mixup"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)