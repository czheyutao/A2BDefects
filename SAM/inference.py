import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import pycocotools.mask as maskUtils
from visualize_utils import visualize_results  # 导入可视化函数
import time

# 数据集类
class CocoInferenceDataset(Dataset):
    """COCO格式推理数据集，保持原始bbox尺寸"""
    def __init__(self, data_root, data_type, prompt_json_path=None):
        annotation_path = os.path.join(data_root, "annotations", f"instances_{data_type}.json")
        self.data_root = data_root
        self.data_type = data_type
        self.coco = COCO(annotation_path)
        self.image_ids = self.coco.getImgIds()
        self.categories = self.coco.loadCats(self.coco.getCatIds())

        if prompt_json_path:
            # 读取 prediction.json 文件
            with open(prompt_json_path, 'r') as f:
                data = json.load(f)
        else:
            # 读取原始的 annotation.json 文件
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            
        if "annotations" in data:
            data = data["annotations"]

        # 创建一个字典，用于存储每个 image_id 对应的所有标注
        result = {}
        for item in data:
            image_id = item['image_id']
            # 复制原始字典并移除 image_id 字段
            new_item = item.copy()
            del new_item['image_id']
            # 将条目添加到对应的 image_id 分组中
            if image_id not in result:
                result[image_id] = []
            result[image_id].append(new_item)
        
        self.prompt_dict = result
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # 加载元数据
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # 加载原始图像
        img_path = os.path.join(self.data_root, self.data_type, img_info["file_name"])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # 使用prompt_dict中的bbox
        annotations = []
        if img_id in self.prompt_dict:
            annotations = self.prompt_dict[img_id].copy()

        return {
            "image_id": img_id,
            "image": image,
            "original_size": (original_h, original_w),
            "annotation": annotations,
            "file_name": img_info["file_name"],
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "image_ids": [x["image_id"] for x in batch],
            "images": [x["image"] for x in batch],
            "annotations": [x["annotation"] for x in batch],
            "original_sizes": [x["original_size"] for x in batch],
            "file_names": [x["file_name"] for x in batch],
        }

def show_points(coords, labels, ax, marker_size=375):
    """
    在指定的matplotlib坐标系中绘制点
    参数:
    coords (np.ndarray): 二维数组，表示点的坐标
    labels (np.ndarray): 一维数组，表示点的标签
    ax (matplotlib.axes.Axes): 用于绘制图形的matplotlib坐标系对象
    marker_size (int, optional): 点的大小，默认为375
    返回值:
    None
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def mask2coco(masks, img_id, categories, scores):
    """
    将mask转换为COCO格式的标注
    参数:
        masks: 预测的mask数组列表，每个元素为(H,W)的bool数组
        imgid: 图像ID
        categories: 对应的类别信息列表
        scores: 对应的置信度列表
    返回值:
        coco_result: COCO格式的标注结果
    """
    coco_output = []
    for mask, category, score in zip(masks, categories, scores):
        # 生成RLE编码
        rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')

        # 从mask计算bbox
        rows, cols = np.where(mask)
        if len(rows) == 0 or len(cols) == 0:
            continue  # 跳过空mask的情况
        
        # 计算COCO格式的bbox [x,y,width,height]
        x_min = cols.min().item()
        y_min = rows.min().item()
        x_max = cols.max().item()
        y_max = rows.max().item()
        bbox = [
            int(x_min), 
            int(y_min), 
            int(x_max - x_min),  # width
            int(y_max - y_min)   # height
        ]

        # 构建COCO标注格式
        coco_result = {
            "image_id": img_id,
            "category_id": category,
            "bbox": bbox,
            "iscrowd": 0,
            "score": score,
            "segmentation": rle
        }
        coco_output.append(coco_result)
    return coco_output
def sam_inference_with_dataloader(
    data_root,
    data_type,
    prompt_json_path=None,
    model_type="vit_b",
    checkpoint_path="/data/hyt/SAM/wz_b/sam_vit_b.pth",
    device="cuda",
    batch_size=4,
    save_dir="/data/hyt/SAM/results",
    save_name="wz", 
):
    # 初始化数据集和数据加载器
    dataset = CocoInferenceDataset(data_root, data_type, prompt_json_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=CocoInferenceDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 初始化SAM模型
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 初始化结果收集列表
    all_results = []

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
            for ann in img_annotations:
                x, y, w, h = ann["bbox"]
                raw_boxes.append([x, y, x + w, y + h])
                category_ids.append(ann["category_id"])
                scores.append(ann["score"] if "score" in ann else 1.0)  # 默认置信度为1.0

            # 限制处理的bbox数量，选择置信度最高的300个
            if len(scores) > 100:
                # 按置信度降序排序
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                # 取前300个
                sorted_indices = sorted_indices[:100]
                raw_boxes = [raw_boxes[i] for i in sorted_indices]
                category_ids = [category_ids[i] for i in sorted_indices]
                scores = [scores[i] for i in sorted_indices]

            # 准备数据
            image = batch["images"][idx]
            
            # 设置图像嵌入
            start_time = time.perf_counter()
            predictor.set_image(image)

            # 处理原始bbox
            boxes = predictor.transform.apply_boxes(np.array(raw_boxes), batch["original_sizes"][idx])
            boxes_tensor = torch.as_tensor(boxes, device=device).unsqueeze(1)

            # 执行预测
            with torch.no_grad():
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_tensor,
                    mask_input=None,
                    multimask_output=False
                )
            # 阈值后处理
            masks = (masks > sam.mask_threshold).squeeze(1).cpu().numpy()
            torch.cuda.synchronize()  # 确保 GPU 计算完成（仅限 CUDA）
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
            print(f"推理时间: {(end_time - start_time) * 1000:.2f} 毫秒")
            #计算平均推理时间

            # 按coco格式保存
            coco_result = mask2coco(masks, batch["image_ids"][idx], category_ids, scores)

            # 将结果添加到总结果列表
            all_results.extend(coco_result.copy())

            # 可视化保存图片
            # os.makedirs(save_dir, exist_ok=True)
            # output_path = f"result_{batch['image_ids'][idx]}.jpg"
            # visualize_results(
            #     batch["images"][idx],
            #     masks,
            #     raw_boxes,
            #     scores,
            #     category_ids,
            #     os.path.join(save_dir, output_path)
            # )

    # 保存json结果
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{data_type}_{save_name}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"推理结果已保存到 {output_path}")

# 主函数改造
if __name__ == "__main__":
    inference_times = []
    sam_inference_with_dataloader(
        data_root="/data/hyt/mmdetection/WZ",
        data_type="test",
        # prompt_json_path="/data/hyt/DEIM/torch_results_coco.json",
        model_type="vit_b",
        checkpoint_path="/data/hyt/SAM/wz_b/step=2500-val_per_mask_iou=0.81.pth",
        device="cuda:7",
        batch_size=4,
        save_name="test"
    )
    # 计算平均推理时间
    avg_inference_time = np.mean(inference_times)
    print(f"平均推理时间: {avg_inference_time * 1000:.2f} 毫秒")