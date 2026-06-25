_base_ = "../m2_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="keepaugment",
    aug_args=dict(
        keepaugment=dict(threshold=0.5, mode='paste', 
                         randaugment_n=2, randaugment_m=9),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)