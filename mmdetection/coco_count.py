import json
import os
import argparse
from collections import defaultdict
import glob
def count_coco_instances(json_path):
    """统计单个COCO标注文件的实例数量"""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    category_map = {c['id']: c['name'] for c in coco_data['categories']}
    instance_counts = defaultdict(int)
    
    for ann in coco_data['annotations']:
        instance_counts[ann['category_id']] += 1
    
    return {
        'total': sum(instance_counts.values()),
        'categories': category_map,
        'counts': dict(instance_counts)
    }

def main():
    parser = argparse.ArgumentParser(description='统计COCO数据集实例数量')
    parser.add_argument('--dir', type=str, required=True,
                      help='包含COCO标注文件的目录路径')
    args = parser.parse_args()

    # 自动检测数据集文件
    file_patterns = {
        'train': '*train*',
        'val': '*val*',
        'test': '*test*'
    }
    
    results = {}
    for dataset, pattern in file_patterns.items():
        # 使用glob查找匹配文件
        matches = list(glob.glob(os.path.join(args.dir, f'*{pattern}*.json')))
        if not matches:
            print(f"[警告] 未找到{dataset}集的标注文件")
            continue
            
        json_path = matches[0]  # 取第一个匹配文件
        print(f"正在处理 {dataset} 集: {os.path.basename(json_path)}")
        
        if stats := count_coco_instances(json_path):
            results[dataset] = stats
        else:
            print(f"[错误] {dataset}集文件读取失败")

    # 汇总统计结果
    if not results:
        print("未找到有效的标注文件")
        return

    # 打印各数据集统计
    for dataset, data in results.items():
        print(f"\n{dataset.upper()} 集统计:")
        print(f"总实例数: {data['total']}")
        print("各类别数量:")
        for cat_id, count in data['counts'].items():
            name = data['categories'][cat_id]
            print(f"  {name}: {count}")

    # 合并所有统计
    merged = defaultdict(int)
    total = 0
    category_names = {}
    
    for data in results.values():
        total += data['total']
        category_names.update(data['categories'])  # 合并类别名称
        for cat_id, count in data['counts'].items():
            merged[cat_id] += count

    # 打印合并结果
    print("\n合并统计:")
    print(f"总实例数: {total}")
    print("各类别总数:")
    for cat_id, total_count in merged.items():
        name = category_names.get(cat_id, f'未知类别_{cat_id}')
        print(f"  {name}: {total_count}")

if __name__ == "__main__":
    main()