_base_ = "../m2_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="starmix",
    starmix=dict(is_vit=False),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)