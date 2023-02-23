_base_ = '../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
classes = ('good', 'bad', 'more_than_two')

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data_root = str("./leaves/")


workflow = [('train', 1), ('val', 1)]

data = dict(
    
    train=dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='CocoDataset',
        ann_file=data_root+'train.json',
        classes= classes,
        img_prefix='./leaves/Leaf Image/',
        pipeline = train_pipeline
    )),
    val=dict(
        type='CocoDataset',
        img_prefix=data_root+'Leaf Image/',
        classes=classes,
        ann_file=data_root+'valid.json'),
    test=dict(
        type='CocoDataset',
        img_prefix=data_root+'Leaf Image/',
        classes=classes,
        ann_file=data_root+'test.json')
        
    )


runner = dict(type='EpochBasedRunner', max_epochs=100)


load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'  # noqa
