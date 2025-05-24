import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict

class BboxSizeAnalyzer:
    def __init__(self, annotation_paths):
        """
        annotation_paths: dict格式 {'train': path, 'val': path, 'test': path}
        """
        self.annotation_paths = annotation_paths
        self.class_areas = defaultdict(list)  # 按类别存储面积
        self.class_thresholds = {}           # 每个类别的阈值
        self.class_ranges = {}               # 每个类别的范围
        self.class_range_labels = ['small', 'medium', 'large']
        self.stats = defaultdict(dict)       # 存储统计结果
        self.class_names = {}                # 类别ID到名称的映射

    def load_all_data(self):
        """加载所有数据集并按类别保存面积"""
        for split, path in self.annotation_paths.items():
            with open(path, 'r') as f:
                coco_data = json.load(f)
            
            # 提取类别名称
            if 'categories' in coco_data:
                for cat in coco_data['categories']:
                    self.class_names[cat['id']] = cat['name']
            
            # 按类别收集面积
            for ann in coco_data['annotations']:
                category_id = ann['category_id']
                area = ann['bbox'][2] * ann['bbox'][3]
                self.class_areas[category_id].append(area)
            
            # 统计每个split的总实例数
            self.stats[split]['raw_count'] = len(coco_data['annotations'])
        
        # 打印类别分布
        print("类别分布:")
        for cid, areas in self.class_areas.items():
            name = self.class_names.get(cid, f"ID {cid}")
            print(f"{name} ({cid}): {len(areas)} 个实例")

    def cluster_class_areas(self, n_clusters=3):
        """对每个类别单独进行K-means聚类，类别3使用1个聚类，类别2使用2个聚类"""
        for category_id, areas in self.class_areas.items():
            # 判断类别调整聚类数量
            current_n_clusters = n_clusters
            if category_id == 2:
                current_n_clusters = 2
            elif category_id == 3:  # 新增类别3处理
                current_n_clusters = 1
            
            areas_array = np.array(areas).reshape(-1, 1)
            log_areas = np.log1p(areas_array)
            
            scaler = StandardScaler()
            kmeans = KMeans(n_init="auto", n_clusters=current_n_clusters, random_state=42)
            kmeans.fit(scaler.fit_transform(log_areas))
            
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            raw_thresholds = sorted(np.expm1(centers.flatten()))
            
            thresholds = [(raw_thresholds[i] + raw_thresholds[i+1]) / 2 
                         for i in range(len(raw_thresholds)-1)]
            
            self.class_thresholds[category_id] = thresholds
            print(f"类别 {category_id} 阈值: {thresholds}")
    
    def generate_class_ranges(self):
        """为每个类别生成区域划分，类别3特殊处理"""
        for cid, thresholds in self.class_thresholds.items():
            if cid == 3:  # 新增类别3处理逻辑
                self.class_ranges[cid] = [[0, 1e5**2]]  # 只分small一类
            elif len(thresholds) == 1:  # 原有两类情况
                self.class_ranges[cid] = [
                    [0, thresholds[0]], [thresholds[0], 1e5**2]
                ]
            else:  # 原有三类情况
                self.class_ranges[cid] = [
                    [0, thresholds[0]], 
                    [thresholds[0], thresholds[1]], 
                    [thresholds[1], 1e5**2]
                ]

    def count_class_instances(self):
        """统计时包含类别3的所有实例为small"""
        split_size_dist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        split_size_ratio = defaultdict(lambda: defaultdict(dict))

        for split, path in self.annotation_paths.items():
            with open(path, 'r') as f:
                coco_data = json.load(f)

            for ann in coco_data['annotations']:
                category_id = ann['category_id']
                area = ann['bbox'][2] * ann['bbox'][3]

                if category_id == 3:  # 直接标记为small
                    size_label = 'small'
                else:
                    thresholds = self.class_thresholds.get(category_id, [])
                    if not thresholds:
                        continue
                    
                    if len(thresholds) == 1:
                        size_label = 'small' if area < thresholds[0] else 'large'
                    else:
                        if area < thresholds[0]:
                            size_label = 'small'
                        elif area < thresholds[1]:
                            size_label = 'medium'
                        else:
                            size_label = 'large'
                
                split_size_dist[split][category_id][size_label] += 1

        for split in split_size_dist:
            for cid in split_size_dist[split]:
                total = sum(split_size_dist[split][cid].values())
                split_size_ratio[split][cid] = {
                    k: f"{v/total:.1%}" for k, v in split_size_dist[split][cid].items()
                }

        self.stats['splits'] = {
            'size_dist': split_size_dist,
            'size_ratio': split_size_ratio
        }
    
    def print_stats(self):
        """打印时包含类别3的统计结果"""
        print("\n=== 类别尺寸分布统计 ===")
        splits = self.annotation_paths.keys()
        
        for split in splits:
            print(f"\n{split}数据集:")
            split_stats = self.stats['splits']
            size_dist = split_stats.get('size_dist', {}).get(split, {})
            
            for cid in sorted(self.class_ranges.keys()):
                name = self.class_names.get(cid, f"ID {cid}")
                dist = size_dist.get(cid, {})
                ratio = split_stats.get('size_ratio', {}).get(split, {}).get(cid, {})
                
                print(f"{name} ({cid}):")
                # 特别处理类别3的标签显示
                labels = ['small'] if cid == 3 else (
                    ['small', 'large'] if len(self.class_ranges[cid]) == 2 
                    else self.class_range_labels
                )
                
                for size in labels:
                    print(f"  {size:>6}: {dist.get(size, 0):>5} instances ({ratio.get(size, '0.0%')})")

    def visualize_class_distribution(self):
        """可视化每个类别的面积分布"""
        for cid, areas in self.class_areas.items():
            plt.figure(figsize=(12, 8))
            thresholds = self.class_thresholds.get(cid, [])
            
            # 绘制直方图
            plt.hist(np.array(areas)/1e4, bins=50, log=True)
            
            # 绘制阈值线
            for t in thresholds:
                plt.axvline(t/1e4, color='r', linestyle='--')
                
            plt.xlabel("Area (1e4 pixels)", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title(f"Class {cid} Area Distribution", fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            plt.savefig(f"class_{cid}_distribution.pdf", format='pdf')
            plt.close()
    
    def run(self, n_clusters=3):
        """完整流程"""
        self.load_all_data()
        self.cluster_class_areas(n_clusters)
        self.generate_class_ranges()
        self.count_class_instances()
        self.visualize_class_distribution()
        self.print_stats()
        
        return self.class_ranges, self.class_range_labels, self.stats
    
# 使用示例
if __name__ == "__main__":
    annotation_paths = {
        'train': "/data/hyt/mmdetection/WZ/annotations/instances_train.json",
        'val': "/data/hyt/mmdetection/WZ/annotations/instances_val.json",
        'test': "/data/hyt/mmdetection/WZ/annotations/instances_test.json"
    }
    
    analyzer = BboxSizeAnalyzer(annotation_paths)
    areaRng, areaRngLbl, stats = analyzer.run()