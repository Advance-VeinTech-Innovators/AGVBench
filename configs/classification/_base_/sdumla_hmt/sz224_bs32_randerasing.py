# dataset settings
data_source_cfg = dict(type='ImageNet')

data_train_list = 'data/meta/sdumla_hmt/train.txt'
data_train_root = 'data/sdumla_hmt/train/'
data_test_list = 'data/meta/sdumla_hmt/test.txt'
data_test_root = 'data/sdumla_hmt/test/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.2172, 0.2172, 0.2172], std=[0.1685, 0.1685, 0.1685])
train_pipeline = [
    dict(type='Resize', size=(224, 224)),  # stretch to square, no crop
    dict(
        type='RandomErasing_numpy',  # before ToTensor and Normalize
        erase_prob=0.25,
        mode='rand', min_area_ratio=0.02, max_area_ratio=1 / 3,
        fill_color=[104, 116, 124], fill_std=[58, 57, 57]),  # RGB
]
test_pipeline = [
    dict(type='Resize', size=(224, 224)),  # stretch to square, no crop
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
