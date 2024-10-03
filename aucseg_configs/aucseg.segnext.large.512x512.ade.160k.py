_base_ = [
    '../configs/segnext/large/segnext.large.512x512.ade.160k.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[dict(type='SquareAUCLoss', num_classes=150, loss_weight=4.0),
                     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)]))

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
        num_classes=150,
        small_index=[
            26, 27, 31, 28, 29, 30, 32, 33, 35, 34, 36, 37, 38, 40, 39, 41, 42,
            43, 44, 45, 51, 47, 46, 49, 50, 48, 52, 55, 56, 53, 57, 54, 59, 60,
            58, 61, 62, 66, 64, 65, 68, 67, 63, 69, 72, 70, 73, 71, 75, 74, 76,
            78, 79, 80, 77, 81, 83, 82, 84, 86, 85, 88, 87, 90, 91, 89, 92, 94,
            95, 93, 98, 101, 97, 96, 102, 100, 99, 103, 108, 111, 105, 104,
            109, 110, 106, 119, 113, 107, 118, 116, 112, 114, 117, 115, 121,
            127, 120, 123, 129, 122, 125, 132, 133, 135, 128, 130, 124, 134,
            126, 137, 131, 136, 140, 145, 141, 139, 138, 144, 142, 143, 146,
            148, 147, 149
        ],
        memory_size=5,
        p_sample=0.05,
        p_resize=0.4)
]

data = dict(
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)))