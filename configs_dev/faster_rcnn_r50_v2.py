_base_ = './faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'


log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
classes = ('good', 'bad', 'more_than_two')

workflow = [('train', 1), ('val', 1)]


dataset_type = 'CocoDataset'
data_root = "./leaves/"

data = dict(
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CocoDataset',
            classes= classes,
            ann_file=data_root+'train.json',
            img_prefix='./leaves/Leaf Image/',
            )),
    val=dict(
        type='CocoDataset',
        classes= classes,
        ann_file=data_root+'valid.json',
        img_prefix='./leaves/Leaf Image/',
        ),
    test=dict(
        type='CocoDataset',
        classes= classes,
        ann_file=data_root+'test.json',
        img_prefix='./leaves/Leaf Image/',
        ))



optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])


runner = dict(type='EpochBasedRunner', max_epochs=100)


evaluation = dict(interval=1, metric='bbox')





load_from = "configs_dev/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth"
