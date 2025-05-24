# coco_ann_path = "/data/hyt/mmdetection/WZ/annotations/instances_test.json"  # 替换为实际路径
# img_dir = "/data/hyt/mmdetection/WZ/test"            # 替换为实际路径
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.hub

class Config:
    coco_ann_path = "/data/hyt/mmdetection/WZ/annotations/instances_test.json"  # 替换为实际路径
    img_dir = "/data/hyt/mmdetection/WZ/test"            # 替换为实际路径
    category_thresholds = {
        1: [1968.4041093704188, 7027.687065900249],
        2: [2547.5538384457755],
        4: [1842.3162048010013, 7309.093595782676]
    }
    target_categories = [1, 2, 4]  # 需要分尺寸处理的类别

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
model.eval()

class InstanceProcessor:
    def __init__(self):
        self.coco = COCO(Config.coco_ann_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _adaptive_crop(self, img, mask):
        """精确裁剪并保持宽高比调整"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # 扩展5%边界
        h, w = img.shape[:2]
        x_pad = int((x_max - x_min) * 0.05)
        y_pad = int((y_max - y_min) * 0.05)
        
        x_min = max(0, x_min - x_pad)
        x_max = min(w-1, x_max + x_pad)
        y_min = max(0, y_min - y_pad)
        y_max = min(h-1, y_max + y_pad)
        
        cropped = img[y_min:y_max+1, x_min:x_max+1]
        # 新增mask处理：将mask外区域置为白色
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        white_bg = np.full_like(cropped, 0)  # 创建白色背景
        cropped = np.where(cropped_mask[..., None], cropped, white_bg)  # 应用mask
        
        return Image.fromarray(cropped).convert('RGB')

    def process_instance(self, ann):
        try:
            img_info = self.coco.loadImgs(ann["image_id"])[0]
            img_path = os.path.join(Config.img_dir, img_info["file_name"])
            
            # 生成掩码并裁剪
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                return None
                
            img = np.array(Image.open(img_path))
            cropped_img = self._adaptive_crop(img, mask)
            if cropped_img is None:
                return None
                
            # 智能调整尺寸
            processed = transforms.functional.resize(cropped_img, 224)
            tensor = self.transform(processed).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model(tensor).squeeze().cpu().numpy()
            return feature
            
        except Exception as e:
            print(f"Error processing annotation {ann['id']}: {str(e)}")
            return None

class FeatureAnalyzer:
    def __init__(self, coco):
        self.coco = coco
        self.feature_store = {}
        # 初始化存储结构
        for cat_id in self.coco.getCatIds():
            if cat_id in Config.target_categories:
                self.feature_store[cat_id] = {
                    "small": [], "medium": [], "large": []
                }
            else:
                self.feature_store[cat_id] = {"all": []}
    
    def _get_size_group(self, cat_id, area):
        thresholds = Config.category_thresholds.get(cat_id, [0, 0])
        if len(thresholds) == 1:
            if area <= thresholds[0]:
                return "small"
            return "large"
        elif len(thresholds) == 2:
            if area <= thresholds[0]:
                return "small"
            elif area <= thresholds[1]:
                return "medium"
            return "large"
    
    def add_feature(self, ann, feature):
        cat_id = ann["category_id"]
        if cat_id in Config.target_categories:
            group = self._get_size_group(cat_id, ann['bbox'][2] * ann['bbox'][3])
            self.feature_store[cat_id][group].append(feature)
        else:
            self.feature_store[cat_id]["all"].append(feature)
    
    def print_statistics(self):
        """打印各组的实例数量统计"""
        print("\n=== 实例数量统计 ===")
        for cat_id, groups in self.feature_store.items():
            cat_name = self.coco.loadCats(cat_id)[0]["name"]
            print(f"\nCategory {cat_id} ({cat_name}):")
            total = 0
            for group, features in groups.items():
                cnt = len(features)
                print(f"  {group:6}: {cnt} instances")
                total += cnt
            print(f"  Total   : {total} instances")
    
    def calculate_metrics(self):
        """计算改进后的指标：余弦聚合度 + 协方差迹"""
        metrics = {}
        for cat_id, groups in self.feature_store.items():
            metrics[cat_id] = {}
            for group, features in groups.items():
                if len(features) < 2:
                    continue
                
                features = np.array(features)
                
                # 改进1：余弦聚合度
                norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
                center = np.mean(norm_features, axis=0)
                cos_sims = np.dot(norm_features, center)
                aggregation = 1 - np.mean(cos_sims)  # 转换为距离
                
                # 改进2：协方差矩阵迹
                cov_matrix = np.cov(features.T)  # 转置使特征为变量
                variance = np.trace(cov_matrix)
                
                metrics[cat_id][group] = {
                    "aggregation": aggregation,
                    "variance": variance
                }
        return metrics

if __name__ == "__main__":
    processor = InstanceProcessor()
    analyzer = FeatureAnalyzer(processor.coco)
    
    # 处理所有标注
    total = len(processor.coco.anns)
    for idx, ann_id in enumerate(processor.coco.anns):
        ann = processor.coco.anns[ann_id]
        if idx % 100 == 0:
            print(f"Processing {idx+1}/{total}...")
        feature = processor.process_instance(ann)
        if feature is not None:
            analyzer.add_feature(ann, feature)
    
    # 输出统计信息
    analyzer.print_statistics()
    
    # 计算并输出新指标
    print("\n=== 特征指标分析 ===")
    metrics = analyzer.calculate_metrics()
    for cat_id, groups in metrics.items():
        cat_name = processor.coco.loadCats(cat_id)[0]["name"]
        print(f"\nCategory {cat_id} ({cat_name}):")
        for group, vals in groups.items():
            print(f"  {group:6} | Aggregation: {vals['aggregation']:.4f} | Variance: {vals['variance']:.4f}")