# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset
data_train_list = 'data/meta/scut1100/train.txt'
data_train_root = 'data/scut1100/train/'
data_test_list = 'data/meta/scut1100/test.txt'
data_test_root = 'data/scut1100/test/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.288, 0.288, 0.288], std=[0.264, 0.264, 0.264])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),  # 0.85
    dict(type='CenterCrop', size=224),
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
