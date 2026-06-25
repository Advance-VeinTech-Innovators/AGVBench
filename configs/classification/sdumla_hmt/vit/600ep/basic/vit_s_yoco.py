_base_ = "../vit_s_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="yoco"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)