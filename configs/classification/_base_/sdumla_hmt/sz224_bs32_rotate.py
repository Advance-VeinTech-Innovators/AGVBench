fixmatch_augment_policies = [
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
]

# dataset settings
data_source_cfg = dict(type='ImageNet')

data_train_list = 'data/meta/sdumla_hmt/train.txt'
data_train_root = 'data/sdumla_hmt/train/'
data_test_list = 'data/meta/sdumla_hmt/test.txt'
data_test_root = 'data/sdumla_hmt/test/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.2172, 0.2172, 0.2172], std=[0.1685, 0.1685, 0.1685])
train_pipeline = [
    dict(type='Resize', size=(224, 224)),
    dict(type='RandAugment',
         policies=fixmatch_augment_policies,
         num_policies=1, total_level=10,
         magnitude_level=7, magnitude_std=0.5,  # 'rand-m7-mstd0.5'
         hparams=dict(
             pad_val=[114, 123, 125], interpolation='bicubic')),
]
test_pipeline = [
    dict(type='Resize', size=(224, 224)), 
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=4,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=128,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
