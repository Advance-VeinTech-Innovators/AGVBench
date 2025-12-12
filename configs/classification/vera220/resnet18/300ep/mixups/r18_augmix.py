_base_ = "../r18_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="augmix",
    mix_args=dict(
        augmix=dict(mixture_depth=-1, mixture_width=3, severity=1,
                    mean=[0.4399, 0.4399, 0.4399], std=[0.114, 0.114, 0.114]),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)