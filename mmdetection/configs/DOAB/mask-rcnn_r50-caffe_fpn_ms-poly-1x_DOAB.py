# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'
num_classes = 4  # 确保与数据集类别数一致
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4)))


# Modify dataset related settings
data_root = './data/WZ/'
metainfo = {
    # 'classes': ('ML', 'DD', 'CR', 'SS', 'EF', 'IR', 'BI', 'MJ'),
    'classes': ("CA","SS","EG","WS")
    # 'classes' : (
    # "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    # "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # "backpack", "umbrella", "handbag", "tie", "suitcase",
    # "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    # "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    # "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    # "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    # "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    # "microwave", "oven", "toaster", "sink", "refrigerator",
    # "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
    # # 'palette': [
    #     (220, 20, 60),
    # ]
}
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train/instances_default.json',
        data_prefix=dict(img='images/train')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val/instances_default.json',
        data_prefix=dict(img='images/val')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test/instances_default.json',
        data_prefix=dict(img='images/test')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/val/instances_default.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test/instances_default.json')

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
