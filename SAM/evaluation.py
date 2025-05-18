# 导入必要的库
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import pycocotools.mask as maskUtils
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from inference import CocoInferenceDataset

# 计算评价指标
def evaluate_segmentation(pred_masks, gt_masks, category_ids):
    # 二值化处理
    pred_masks = (pred_masks > 0).astype(int)
    gt_masks = (gt_masks > 0).astype(int)

    # 将掩码按位或合并
    pred_masks = np.logical_or.reduce(pred_masks, axis=0)
    gt_masks = np.logical_or.reduce(gt_masks, axis=0)

    # print(pred_masks.shape)
    # print(gt_masks.shape)
    # 计算评价指标
    gt_masks = np.array(gt_masks).reshape(-1)
    pred_masks = np.array(pred_masks).reshape(-1)

    precision = precision_score(gt_masks, pred_masks, average="binary",zero_division=0)
    recall = recall_score(gt_masks, pred_masks, average="binary", zero_division=0)
    f1 = f1_score(gt_masks, pred_masks, average="binary", zero_division=0)
    accuracy = accuracy_score(gt_masks, pred_masks)
    miou = jaccard_score(gt_masks, pred_masks, average="binary", zero_division=0)
    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'mIoU': miou
    }

# 推理函数（原代码简化）
def sam_inference_and_eval(data_root, data_type, prompt_json_path=None, checkpoint_path="/data/hyt/SAM/wz_b/sam_vit_b.pth", model_type="vit_b", device="cuda", batch_size=4):
    # 数据集初始化
    # 初始化数据集和数据加载器
    dataset = CocoInferenceDataset(data_root, data_type, prompt_json_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=CocoInferenceDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    all_results = []
    metrics = []

    # 批量推理
    for batch in tqdm(loader):
        # 逐样本处理
        for idx in range(len(batch["images"])):
            # 获取对应标注信息
            img_annotations = batch["annotations"][idx]
              
            # 提取bbox坐标
            raw_boxes = []
            category_ids = []
            scores = []
            gt_masks = []
            for ann in img_annotations:
                # box resize
                x, y, w, h = ann["bbox"]
                x = x * 512 / batch["original_sizes"][idx][0]
                y = y * 512 / batch["original_sizes"][idx][1]
                w = w * 512 / batch["original_sizes"][idx][0]
                h = h * 512 / batch["original_sizes"][idx][1]
                
                raw_boxes.append([x, y, x + w, y + h])
                category_ids.append(ann["category_id"])
                scores.append(ann["score"] if "score" in ann else 1.0)  # 默认置信度为1.0
                
                # 创建一个空白图像（黑色背景）
                binary_image = np.zeros((512, 512), dtype=np.uint8)
                # 类似box 一样，将多边形坐标点进行缩放
                # 将多边形坐标点缩放到512x512的图像大小
                ann["segmentation"] = np.array(ann["segmentation"]).reshape(-1, 2)
                ann["segmentation"] = [ann["segmentation"]]
                ann["segmentation"][0][:, 0] = ann["segmentation"][0][:, 0] * 512 / batch["original_sizes"][idx][0]
                ann["segmentation"][0][:, 1] = ann["segmentation"][0][:, 1] * 512 / batch["original_sizes"][idx][1]
                ann["segmentation"] = [ann["segmentation"][0]]
                ann["segmentation"] = ann["segmentation"][0]

                # 将多边形坐标点转换为 numpy 数组，并重塑为 (n, 1, 2) 的形状，这是 OpenCV 要求的格式
                pts = np.array(ann["segmentation"], dtype=np.int32).reshape((-1, 1, 2))
                # 在空白图像上绘制并填充多边形，颜色为 255（白色）
                cv2.fillPoly(binary_image, [pts], 255)
                gt_masks.append(binary_image)
            gt_masks = np.array(gt_masks).reshape(-1, 512, 512)

            # # 限制处理的bbox数量，选择置信度最高的300个
            # if len(scores) > 100:
            #     # 按置信度降序排序
            #     sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            #     # 取前300个
            #     sorted_indices = sorted_indices[:100]
            #     raw_boxes = [raw_boxes[i] for i in sorted_indices]
            #     category_ids = [category_ids[i] for i in sorted_indices]
            #     scores = [scores[i] for i in sorted_indices]

            # 准备数据
            image = batch["images"][idx]
            image = cv2.resize(image, (512, 512))
            # 设置图像嵌入
            predictor.set_image(image)

            # 处理原始bbox
            boxes = predictor.transform.apply_boxes(np.array(raw_boxes), (512, 512))
            boxes_tensor = torch.as_tensor(boxes, device=device).unsqueeze(1)

            # 执行预测
            with torch.no_grad():
                pred_masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_tensor,
                    mask_input=None,
                    multimask_output=False
                )
            # 阈值后处理
            pred_masks = (pred_masks > sam.mask_threshold).squeeze(1).cpu().numpy()
            # print(gt_masks.shape)
            # print(pred_masks.shape)
            # #将gt_masks[0]和pred_masks[0]保存为图片
            # cv2.imwrite("gt_masks.png", gt_masks[0])
            # cv2.imwrite("pred_masks.png", pred_masks[0]*255)
            
            # # 保存gt和pred_masks
            
            metric = evaluate_segmentation(pred_masks, gt_masks, category_ids)
            print(metric)
            metrics.append(metric)
            all_results.append((batch['image_ids'][idx], metric))
    return all_results, metrics

if __name__ == "__main__":
    results, metrics = sam_inference_and_eval(
        data_root="/data/hyt/mmdetection/QZ_test",
        data_type="test",
        model_type="vit_b",
        device="cuda:3",
        batch_size=4,
        # checkpoint_path="/data/hyt/SAM/wz_b/step=2500-val_per_mask_iou=0.81.pth"
    )
    print("平均指标:", {k: np.mean([m[k] for m in metrics]) for k in metrics[0]})
    # 保存每个的平均指标
    with open("metrics1.json", "w") as f:
        json.dump({"average_metrics": {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}}, f)
    # 保存每个的指标
    with open("results1.json", "w") as f:
        json.dump({"results": results}, f)

    results, metrics = sam_inference_and_eval(
        data_root="/data/hyt/mmdetection/QZ_test",
        data_type="test",
        model_type="vit_b",
        device="cuda:3",
        batch_size=4,
        checkpoint_path="/data/hyt/SAM/wz_b/step=2500-val_per_mask_iou=0.81.pth"
    )
    print("平均指标:", {k: np.mean([m[k] for m in metrics]) for k in metrics[0]})
    # 保存每个的平均指标
    with open("metrics2.json", "w") as f:
        json.dump({"average_metrics": {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}}, f)
    # 保存每个的指标
    with open("results2.json", "w") as f:
        json.dump({"results": results}, f)
            
