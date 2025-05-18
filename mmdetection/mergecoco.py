import json
import os
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

def merge_coco_datasets(dataset1_path, dataset2_path, output_path, splits=['train', 'val', 'test']):
    """合并两个COCO格式数据集"""
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)
    
    for split in splits:
        # 初始化合并后的数据结构
        merged = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [],
            "info": {}
        }
        
        # 处理第一个数据集
        process_dataset(dataset1_path, split, merged, output_path)
        # 处理第二个数据集
        process_dataset(dataset2_path, split, merged, output_path)
        
        # 保存合并后的JSON
        with open(f'{output_path}/annotations/instances_{split}.json', 'w') as f:
            json.dump(merged, f, indent=2)

def process_dataset(dataset_path, split, merged, output_path):
    """处理单个数据集"""
    # 加载原始标注
    coco = COCO(f'{dataset_path}/annotations/instances_{split}.json')
    
    # 映射表：旧ID -> 新ID
    img_id_map = {}
    cat_id_map = {}
    
    # ========== 合并图片 ==========
    for img in coco.dataset['images']:
        # 生成唯一新文件名（防止冲突）
        new_file_name = f"{os.path.basename(dataset_path)}_{img['file_name']}"
        new_img_id = len(merged['images']) + 1
        
        # 复制图片到新目录
        src_path = os.path.join(dataset_path, split, img['file_name'])
        dst_path = os.path.join(output_path, split, new_file_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
        
        # 更新图片记录
        img_id_map[int(img['id'])] = new_img_id
        img['id'] = new_img_id
        img['file_name'] = new_file_name
        merged['images'].append(img)
        
    
    # ========== 合并类别 ==========
    for cat in coco.dataset['categories']:
        # 检查是否已存在同名类别
        existing_cat = next((c for c in merged['categories'] if c['name'] == cat['name']), None)
        if existing_cat:
            cat_id_map[cat['id']] = existing_cat['id']
        else:
            new_cat_id = len(merged['categories']) + 1
            cat_id_map[cat['id']] = new_cat_id
            cat['id'] = new_cat_id
            merged['categories'].append(cat)
            
    
    # ========== 合并标注 ==========
    for ann in tqdm(coco.dataset['annotations'], desc=f"Processing {split} annotations"):
        new_ann_id = len(merged['annotations']) + 1
        ann['id'] = new_ann_id
        # print(img_id_map)
        ann['image_id'] = img_id_map[ann['image_id']]
        ann['category_id'] = cat_id_map[ann['category_id']]
        merged['annotations'].append(ann)

# 使用示例
merge_coco_datasets(
    dataset1_path='/data/hyt/mmdetection/WZ',
    dataset2_path='/data/hyt/mmdetection/QZ',
    output_path='/data/hyt/mmdetection/WZQZ',
)