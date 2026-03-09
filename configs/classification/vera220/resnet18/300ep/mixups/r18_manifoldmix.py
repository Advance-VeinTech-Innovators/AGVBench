_base_ = "../r18_mixups_sz224_bs32.py"

model = dict(
    alpha=1.0,
    mix_mode="mixup",
    backbone=dict(
        type='ResNet_Mix', 
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)