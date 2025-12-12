_base_ = "../ampvnet_basicaug_sz224_bs32.py"

model = dict(
    alpha=0.2,
    mix_mode="saliencymix"
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)