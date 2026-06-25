_base_ = "../m2_basicaug_sz224_bs32.py"

model = dict(
    aug_mode="smdwt_pca",
    aug_args=dict(
        smdwt_pca=dict(thresholds=(0.55, 0.65), wavelet=('bior1.3', 'bior4.4', 'bior6.8')),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)