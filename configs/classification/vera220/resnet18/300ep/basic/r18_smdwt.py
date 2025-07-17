_base_ = "../r18_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="smdwt_pca",
    aug_args=dict(
        smdwt_pca=dict(wavelet='5/3'),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)