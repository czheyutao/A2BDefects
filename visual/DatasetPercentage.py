import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from collections import defaultdict
import pandas as pd

# ----------------------------
# 1. 定义数据集路径及偏移量（确保图片ID唯一）
# ----------------------------
split_config = {
    'train': {
        'ann_path': '/data/hyt/mmdetection/WZ/annotations/instances_train.json',  # 替换为实际路径
        'id_offset': 0                  # 训练集ID偏移量
    },
    'val': {
        'ann_path': '/data/hyt/mmdetection/WZ/annotations/instances_val.json',    # 替换为实际路径
        'id_offset': 1000000            # 验证集ID偏移量（假设训练集图片数 < 1e6）
    },
    'test': {
        'ann_path': '/data/hyt/mmdetection/WZ/annotations/instances_test.json',   # 替换为实际路径
        'id_offset': 2000000            # 测试集ID偏移量
    }
}

# ----------------------------
# 2. 初始化统计存储结构
# ----------------------------
category_images = defaultdict(set)       # 记录每个类别出现的图片ID（全局唯一）
category_pixels = defaultdict(lambda: defaultdict(float))  # 记录像素占比
cat_id_to_name = {}                     # 类别ID到名称的映射
total_images = 0                        # 总图片数

# ----------------------------
# 3. 遍历所有数据集（train/val/test）
# ----------------------------
for split in split_config:
    # 加载当前数据集
    ann_path = split_config[split]['ann_path']
    id_offset = split_config[split]['id_offset']
    coco = COCO(ann_path)
    
    # 获取类别信息（仅第一次加载时记录）
    if not cat_id_to_name:
        categories = coco.loadCats(coco.getCatIds())
        cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # 处理当前数据集的所有图片
    img_ids = coco.getImgIds()
    total_images += len(img_ids)  # 累加总图片数
    
    for img_id in img_ids:
        # 生成全局唯一的图片ID
        global_img_id = img_id + id_offset
        
        # 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        width, height = img_info['width'], img_info['height']
        total_pixels = width * height
        
        # 加载当前图片的标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # 按类别分组标注
        cat_anns = defaultdict(list)
        for ann in annotations:
            cat_anns[ann['category_id']].append(ann)
        
        # 处理每个类别的像素
        for cat_id, anns in cat_anns.items():
            # 记录包含该类别的图片ID（使用全局ID）
            category_images[cat_id].add(global_img_id)
            
            # 合并掩码计算像素占比
            merged_mask = np.zeros((height, width), dtype=np.uint8)
            for ann in anns:
                mask = coco.annToMask(ann)
                merged_mask = np.logical_or(merged_mask, mask)
            
            pixel_perc = np.sum(merged_mask) / total_pixels * 100
            category_pixels[cat_id][global_img_id] = pixel_perc

# ----------------------------
# 4. 计算合并后的统计指标
# ----------------------------
stats = []
for cat_id in category_images:
    cat_name = cat_id_to_name.get(cat_id, f'Unknown_{cat_id}')
    
    # 计算百分比
    perc_images = len(category_images[cat_id]) / total_images * 100
    avg_pixel_perc = np.mean(list(category_pixels[cat_id].values()))
    
    stats.append({
        'Damage Type': cat_name,
        'Percentage of Images (%)': perc_images,
        'Average Percentage of Pixels (%)': avg_pixel_perc
    })

# 转换为 DataFrame
df = pd.DataFrame(stats)

# ----------------------------
# 5. 绘图函数（修改：添加保存为PDF的功能）
# ----------------------------
def plot_stat(df, y_column, title, save_path=None):
    # 按指定列排序
    df_sorted = df.sort_values(by=y_column, ascending=False)
    
    # 定义缩写到完整名称的映射
    damage_type_mapping = {
        'CA': 'Color Aberration',
        'SS': 'Surface Spalling',
        'EG': 'Excessive Gap',
        'WS': 'Water Stain'
    }
    
    # 替换 X 轴标签为完整名称
    df_sorted['Damage Type'] = df_sorted['Damage Type'].map(damage_type_mapping)
    
    plt.figure(figsize=(6, 6))
    bars = plt.bar(
        df_sorted['Damage Type'],
        df_sorted[y_column],
        color=plt.cm.tab20.colors[:len(df_sorted)],
        width=0.6,
    )
    # 添加背景横线
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')
    
    plt.xlabel('Damage Type', fontsize=12)
    plt.ylabel(y_column, fontsize=12)  # 添加 y 轴标签
    plt.xticks(ticks=range(len(df_sorted)), labels=[])  # 修改：将 X 轴标签设置为空列表
    plt.legend(bars, df_sorted['Damage Type'], title='', loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300)  # 修改：将保存格式改为PDF
    plt.show()

# 绘制合并后的统计图
# 保存 Percentage of Images 图像为PDF
plot_stat(df, 'Percentage of Images (%)', 'Percentage of Images (%)', save_path='images_percentage.pdf')

# 保存 Average Pixel Coverage 图像为PDF
plot_stat(df, 'Average Percentage of Pixels (%)', 'Average Pixel Coverage (%)', save_path='pixel_coverage.pdf')
