_base_ = [
    '../configs/segnext/large/segnext.large.1024x1024.city.160k.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[dict(type='SquareAUCLoss', num_classes=19, loss_weight=1.0),
                     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)]))

dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    dict(
        type='TMemoryBank',
        num_classes=19,
        small_index=[6, 12, 14, 15, 16, 17, 18],
        memory_size=5,
        p_sample=1,
        p_resize=1)
]

data = dict(
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=train_pipeline)))