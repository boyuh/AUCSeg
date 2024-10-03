_base_ = [
    '../configs/segnext/large/segnext.large.512x512.coco_stuff164k.80k.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[dict(type='SquareAUCLoss', num_classes=171, loss_weight=1.0),
                     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)]))

dataset_type = 'COCOStuffDataset'
data_root = 'data/coco_stuff164k'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    dict(
        type='TMemoryBank',
        num_classes=171,
        small_index=[
            3, 20, 97, 89, 87, 166, 16, 116, 127, 72, 131, 93, 55, 69, 63, 17,
            13, 121, 62, 4, 23, 22, 77, 61, 95, 138, 28, 98, 48, 25, 135, 144,
            117, 113, 162, 19, 8, 146, 170, 41, 58, 80, 115, 88, 132, 46, 163,
            82, 158, 107, 18, 151, 21, 126, 50, 54, 71, 73, 168, 152, 75, 140,
            104, 14, 39, 1, 139, 150, 66, 118, 49, 74, 110, 52, 83, 37, 81, 10,
            11, 149, 47, 26, 92, 102, 67, 96, 68, 24, 40, 154, 124, 134, 156,
            51, 12, 9, 33, 125, 36, 38, 129, 148, 65, 141, 43, 31, 76, 29, 30,
            91, 27, 42, 119, 44, 64, 35, 122, 79, 34, 32, 167, 70, 78
        ],
        memory_size=5,
        p_sample=0.05,
        p_resize=0.1)
]

data = dict(
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train2017',
            ann_dir='annotations/train2017',
            pipeline=train_pipeline)))