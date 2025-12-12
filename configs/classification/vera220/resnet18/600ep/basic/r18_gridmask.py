_base_ = "../r18_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="gridmask",
    aug_args=dict(
        gridmask=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
                    cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)