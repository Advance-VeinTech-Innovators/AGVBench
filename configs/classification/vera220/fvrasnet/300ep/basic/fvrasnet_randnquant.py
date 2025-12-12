_base_ = "../fvrasnet_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="randnquant"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)