# dataset settings
dataset_type = 'MultiRepDataset'
data_root = "/mnt/tbdisk/tangyin/MultiCounter/MRepData/"
clip_length = 64
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_period=True,
        with_id=True),
    dict(type='Resize', img_scale=[(224, 224)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_periodicity', 'gt_periods', 'gt_ids'])]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])]
data = dict(
    samples_per_gpu=4,   # batch_size
    workers_per_gpu=8,   # num_workers
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        clip_length=clip_length,
        img_prefix=data_root + 'train_raw_frames/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        clip_length=clip_length,
        img_prefix=data_root + 'val_raw_frames/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        clip_length=clip_length,
        img_prefix=data_root + 'test_raw_frames/',
        pipeline=test_pipeline))
evaluation = dict(metric=['segm'])
