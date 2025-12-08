_base_ = "../starlknet_b_mixups_sz224_bs32.py"

model = dict(
    alpha=2.0,
    mix_mode="manifoldmix",
    backbone=dict(
        type='ResNet_Mix', 
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)