# The new config inherits a base config to highlight the necessary modification
_base_ = 'yolov3_mobilenetv2_320_300e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=3))

# Modify dataset related settings
dataset_type = 'COCODataset'

classes = ('more_than_two','good','bad')


data_root = str("./leaves/")


workflow = [('train', 1), ('val', 1)]

data = dict(
    
    train=dict(
    type='RepeatDataset',
        times=1,
    dataset=dict(
        img_prefix=data_root+'Leaf Image/',
        classes=classes,
        ann_file=data_root+'train.json')),
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

# evaluation = dict(interval=2, metric=['bbox'])
runner = dict(type='EpochBasedRunner', max_epochs=100)


# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'

