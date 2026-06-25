_base_ = "../m2_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="ricap",
    aug_args=dict(
        ricap=dict(choose_num=2, ),
    ),
    head=dict(
        type='ClsMixupHead',  # default CE
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
