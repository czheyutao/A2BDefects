import os
import cv2
from pycocotools.coco import COCO
from visualize_utils import visualize_results
import json
import matplotlib.pyplot as plt

def visualize_single_image(
    coco, img_id, save_dir, model_name, show_box_label=False, show_mask_label=False, nms_label=False
):
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
    output_path = os.path.join(
        save_dir, f"result_{model_name}_{img_id}"
    )  # 修改文件名，包含模型名称
    visualize_results(
        image,
        masks,
        raw_boxes,
        scores,
        category_ids,
        output_path,
        show_box_label=show_box_label,
        show_mask_label=show_mask_label,
        nms_label=nms_label,
    )


if __name__ == "__main__":
    # 数据集路径
    data_root = "/data/hyt/mmdetection/WZ"
    data_type = "train"
    gt_path = os.path.join(data_root, "annotations", f"instances_{data_type}.json")

    # 新增分类目录配置
    category_map = {
        # "intricate": [8, 22, 114, 318],
        # "simple": [51, 1150, 1143],
        # "light": [191, 206, 68],
        # "dark": [704, 615, 440, 248],
        # "occlusion": [294, 314, 355, 407, 808],
        # "withoutoocclusion": [938, 939, 957],
        # "viewopen": [1144, 1147, 1100],
        # "viewclose": [75, 76, 96, 98]
        # 'ws':[427, 428],
        # ``'ca':[613, 570],
        # 'eg':[189, 145],
        # 'ss':[453, 481]
    }

    # 支持选择不同模型的JSON结果文件
    resFiles = {
        "ConvNeXT": "/data/hyt/mmdetection/convnext100/coco_instance/test.segm.json",
        "MaskRCNN": "/data/hyt/mmdetection/maskrcnn/coco_instance/test.segm.json",
        "Mask2Former": "/data/hyt/mmdetection/mask2former/coco_instance/test.segm.json",
        "Co-DETR": "/data/hyt/Co-DETR-main/condino.segm100.json",
        "YOLO-v8": "/data/hyt/yolo/yoloruns/WZ/test-v8l-seg-no/predictions.json",
        "YOLO-v9": "/data/hyt/yolo/yoloruns/WZ/test-v9c-seg-no/predictions.json",
        "YOLO-v10": "/data/hyt/yolo/yoloruns/WZ/test-v10l-seg-no/predictions.json",
        "YOLO-v11": "/data/hyt/yolo/yoloruns/WZ/test-11l-seg-no/predictions.json",
        "YOLOv11+SAM": "/data/hyt/SAM/results/test_yolo11_sam.json",
        "DEIM+SAM": "/data/hyt/SAM/results/test_deim_sam.json",
        "ConvNeXT+SAM": "/data/hyt/SAM/results/test_convnext_sam.json",
        "Co-DETR+SAM": "/data/hyt/SAM/results/test_codi-no_sam.json",
    }

    coco = COCO(gt_path)
    all_img_ids = coco.getImgIds()

    # 修改可视化循环结构
    for category, img_ids in category_map.items():
        for img_id in img_ids:
            save_dir = f"/data/hyt/mmdetection/visual/quality_image_results/{category}/{img_id}"
            os.makedirs(save_dir, exist_ok=True)

            # 原图保存逻辑保持不变，路径包含分类目录
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(data_root, data_type, img_info["file_name"])
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]
            dpi = 600
            figsize = (W / dpi * 4, H / dpi * 4)

            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(
                os.path.join(save_dir, f"ori_{img_id}.pdf"),
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
            )
            plt.close()

            # 可视化标注结果
            visualize_single_image(
                coco,
                int(img_id),
                save_dir,
                f"gt_mask",
                # show_box_label=True,
                show_mask_label=True,
            )

    # 模型可视化部分同步修改
    for model_name, annotation_path in resFiles.items():
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        coco = coco.loadRes(data)
        
        for category, img_ids in category_map.items():
            for img_id in img_ids:
                save_dir = f"/data/hyt/mmdetection/visual/quality_image_results/{category}/{img_id}"
                os.makedirs(save_dir, exist_ok=True)
                visualize_single_image(coco, int(img_id), save_dir, model_name, 
                                     show_box_label=True, show_mask_label=True, 
                                     nms_label=True)