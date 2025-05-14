import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from visualize_utils import visualize_results
import json
def visualize_single_image(coco, img_id, save_dir, model_name, show_box=True):
    """
    可视化单张图片的SAM预测结果并保存
    参数:
        coco: COCO对象
        img_id: 图片ID
        save_dir: 结果保存目录
        model_name: 模型名称
    """
    # 加载图片信息
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(data_root, data_type, img_info["file_name"])
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # 加载标注信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # 提取bbox坐标、类别ID和置信度
    raw_boxes = []
    category_ids = []
    scores = []
    masks = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        raw_boxes.append([x, y, x + w, y + h])
        category_ids.append(ann["category_id"])
        scores.append(ann["score"] if "score" in ann else 1.0)
        # 将RLE编码的mask转换为二值掩码
        mask = coco.annToMask(ann)
        masks.append(mask.astype(bool))

    # 可视化保存图片
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"result_{model_name}_{img_id}")  # 修改文件名，包含模型名称
    visualize_results(
        image,
        masks,
        raw_boxes,
        scores,
        category_ids,
        output_path,
        show_box_label=show_box,
    )

if __name__ == "__main__":
    # 数据集路径
    data_root = "/data/hyt/mmdetection/WZ"
    data_type = "test"
    gt_path = os.path.join(data_root, "annotations", f"instances_{data_type}.json")
    
    # 支持选择不同模型的JSON结果文件
    resFiles = {
        'ConvNeXT': '/data/hyt/mmdetection/convnext100/coco_instance/test.segm.json',
        'MaskRCNN': '/data/hyt/mmdetection/maskrcnn/coco_instance/test.segm.json',
        'Mask2Former': '/data/hyt/mmdetection/mask2former/coco_instance/test.segm.json',
        'Co-DETR': '/data/hyt/Co-DETR-main/condino.segm100.json',
        'YOLO-v8': '/data/hyt/yolo/yoloruns/WZ/test-v8l-seg-no/predictions.json',
        'YOLO-v9': '/data/hyt/yolo/yoloruns/WZ/test-v9c-seg-no/predictions.json',
        'YOLO-v10': '/data/hyt/yolo/yoloruns/WZ/test-v10l-seg-no/predictions.json',
        'YOLO-v11': '/data/hyt/yolo/yoloruns/WZ/test-11l-seg-no/predictions.json',
        'YOLOv11+SAM': '/data/hyt/SAM/results/test_yolo11_sam.json',
        'DEIM+SAM': '/data/hyt/SAM/results/test_deim_sam.json',
        'ConvNeXT+SAM': '/data/hyt/SAM/results/test_convnext_sam.json',
        'Co-DETR+SAM': '/data/hyt/SAM/results/test_codino_sam.json'
    }

    selected_img_ids = []  # 存储随机选择的五张图片ID
    coco = COCO(gt_path)
    all_img_ids = coco.getImgIds()
    # selected_img_ids = np.random.choice(all_img_ids, size=10, replace=False)
    # selected_img_ids = [int(img_id) for img_id in selected_img_ids]
    selected_img_ids = [698, 90]
    # 可视化dt
    # 遍历五张图片进行可视化
    for img_id in selected_img_ids:
        save_dir = f"/data/hyt/mmdetection/visual/image_results/{img_id}"
        os.makedirs(save_dir, exist_ok=True)
        visualize_single_image(coco, int(img_id), save_dir, "gt",)

    # # 遍历每个模型并进行可视化
    # for model_name, annotation_path in resFiles.items():
    #     # 初始化COCO对象
    #     with open(annotation_path, 'r') as f:
    #         data = json.load(f)

    #     coco = coco.loadRes(data)

    #     # 如果是第一个模型，随机选择五张图片
    #     # 遍历五张图片进行可视化
    #     for img_id in selected_img_ids:
    #         save_dir = f"/data/hyt/SAM/model_image_results/{img_id}"
    #         os.makedirs(save_dir, exist_ok=True)
    #         visualize_single_image(coco, int(img_id), save_dir, model_name)
