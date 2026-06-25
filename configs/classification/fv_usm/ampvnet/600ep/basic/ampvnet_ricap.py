_base_ = "../ampvnet_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="ricap",
    aug_args=dict(
        ricap=dict(choose_num=2),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)