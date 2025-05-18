import json
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 加载并合并所有数据集
base_path = '/data/hyt/mmdetection/WZ/annotations/'
datasets = ['train', 'val', 'test']

# 存储原始数据集结构和合并后的数据
coco_data_dict = {}
all_images = []
all_annotations = []

for dataset in datasets:
    with open(f'{base_path}instances_{dataset}.json', 'r') as f:
        data = json.load(f)
        coco_data_dict[dataset] = data
        all_images.extend(data['images'])
        all_annotations.extend(data['annotations'])

# 2. 构建特征矩阵
image_anns = defaultdict(list)
for ann in all_annotations:
    image_anns[ann['image_id']].append(ann)

features = []
image_id_map = {}  # 保存image_id到特征索引的映射
for idx, img_info in enumerate(all_images):
    img_id = img_info['id']
    anns = image_anns.get(img_id, [])
    
    # 计算标注数量
    ann_count = len(anns)
    
    # 计算面积占比
    total_area = sum(bbox[2]*bbox[3] for bbox in (ann['bbox'] for ann in anns))
    width = img_info['width']
    height = img_info['height']
    area_ratio = total_area / (width * height) if (width * height) > 0 else 0
    
    features.append([ann_count, area_ratio])
    image_id_map[img_id] = idx  # 记录image_id到特征索引的映射

# 3. 特征标准化和聚类
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(scaled_features)

# 4. 确定hard cluster
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
hard_cluster = 0 if sum(cluster_centers[0]) > sum(cluster_centers[1]) else 1

# 5. 创建image_id到cluster标签的映射
image_id_to_label = {img['id']: labels[image_id_map[img['id']]] for img in all_images}

# 6. 为每个原始数据集创建easy/hard分割
for dataset in datasets:
    original_data = coco_data_dict[dataset]
    
    # 创建新的数据结构
    hard_data = {
        "images": [],
        "annotations": [],
        "categories": original_data['categories']
    }
    easy_data = {
        "images": [],
        "annotations": [],
        "categories": original_data['categories']
    }
    
    # 构建当前数据集的image到annotations映射
    current_ann_map = defaultdict(list)
    for ann in original_data['annotations']:
        current_ann_map[ann['image_id']].append(ann)
    
    # 分配图片到对应的数据集
    for img_info in original_data['images']:
        img_id = img_info['id']
        label = image_id_to_label.get(img_id)
        
        if label == hard_cluster:
            target = hard_data
        else:
            target = easy_data
        
        # 添加图片信息
        target['images'].append(img_info)
        # 添加对应的annotations
        if img_id in current_ann_map:
            target['annotations'].extend(current_ann_map[img_id])
    
    # 保存结果
    for difficulty, data in [('hard', hard_data), ('easy', easy_data)]:
        with open(f'{base_path}instances_{dataset}_{difficulty}.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f'{dataset} 分割完成:')
    print(f'  Hard: {len(hard_data["images"])} images, {len(hard_data["annotations"])} annotations')
    print(f'  Easy: {len(easy_data["images"])} images, {len(easy_data["annotations"])} annotations')

# 7. 可视化（可选）
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(
    [f[0] for f in features],
    [f[1] for f in features],
    c=labels,
    cmap='viridis',
    alpha=0.6
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='red',
    s=300,
    marker='X',
    label='Cluster Centers'
)

plt.title('Combined Dataset Clustering')
plt.xlabel('Annotation Count')
plt.ylabel('Area Ratio')
plt.legend()
plt.savefig(f'{base_path}combined_clustering.png')
plt.show()