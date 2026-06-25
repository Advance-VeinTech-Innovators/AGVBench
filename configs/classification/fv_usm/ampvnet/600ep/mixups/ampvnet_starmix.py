_base_ = "../ampvnet_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="starmix",
    mix_args=dict(
        starmix=dict(),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)