_base_ = "../ampvnet_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="randomblur"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)