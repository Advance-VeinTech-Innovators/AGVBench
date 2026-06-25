_base_ = "../starlknet_s_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="guidedmix",
    mix_args=dict(
        guidedmix=dict(
            guided_type='ap',       # 'ap': gradient-based saliency, 'sr': spectral residual
            condition='greedy',     # greedy one-cycle cover for sample pairing
            distance_metric='l2',  # distance metric for pairing: 'l2', 'l1', 'cosine'
            size=(7, 7),            # gaussian blur kernel size for feature smoothing
            sigma=(3, 3),          # gaussian blur sigma
        ),
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)