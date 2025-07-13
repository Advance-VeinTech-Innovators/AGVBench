_base_ = "../ampvnet_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="spnoise",
    aug_args=dict(
        spnoise=dict(prob=0.1, noise_type='random'),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)