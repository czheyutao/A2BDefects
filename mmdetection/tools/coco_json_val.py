from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import json  # 新增导入
import os

# 模型名称
model_names = ["maskrcnn", "mask2former", "convnext100", "codino100"]  # 这里可以替换为你的模型名称
for model_name in model_names:
    types = ["_easy", "_hard", ""]
    eval_type = "segm"
    for type in types:
        # 保存到 CSV 文件
        os.makedirs('/data/hyt/SAM/results', exist_ok=True)
        output_path = f"/data/hyt/SAM/results/test-{model_name}{type}.csv"

        # 定义路径（根据你的实际路径调整）
        image_dir = "/data/hyt/mmdetection/WZ/test"  # Image directory
        gt_json_path = f"/data/hyt/mmdetection/WZ/annotations/instances_test{type}.json"  # Ground truth JSON

        # 加载 Ground Truth (gt)
        coco_gt = COCO(gt_json_path)

        # +++ 新增过滤逻辑 +++

        # 加载原始检测结果文件
        dt_path = f"/data/hyt/mmdetection/{model_name}/coco_instance/test.segm.json"
        if model_name == "codino100":
            dt_path = f"/data/hyt/Co-DETR-main/condino.segm100.json"
        if model_name == "deim":
            dt_path = f"/data/hyt/DEIM/torch_results_coco.json"
        with open(dt_path, 'r') as f:
            dt_data = json.load(f)

        # 获取所有真实图片ID
        valid_img_ids = set(coco_gt.getImgIds())

        # 过滤检测结果（兼容两种常见格式）
        if isinstance(dt_data, dict) and 'annotations' in dt_data:
            # COCO格式的字典结构
            dt_annotations = dt_data['annotations']
        else:
            # 纯注解列表结构
            dt_annotations = dt_data

        # 过滤掉无效图片ID的检测结果
        filtered_dt = [ann for ann in dt_annotations if ann['image_id'] in valid_img_ids]

        print(f"原始检测结果数: {len(dt_annotations)}, 过滤后有效结果数: {len(filtered_dt)}")

        # 加载过滤后的检测结果 (dt)
        coco_dt = coco_gt.loadRes(filtered_dt)
        # +++ 过滤结束 +++

        # 初始化 COCOeval 用于分割评估
        coco_eval = COCOeval(coco_gt, coco_dt, eval_type)

        # 获取类别ID和名称的映射
        cat_ids = coco_gt.getCatIds()
        cats = coco_gt.loadCats(cat_ids)
        cat_names = [cat['name'] for cat in cats]

        # 准备数据存储
        results = []

        # 先计算整体结果
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results.append({
            'Category': 'all',
            'AP@0.5:0.95': coco_eval.stats[0],
            'AP@0.5': coco_eval.stats[1],
            'AP@0.75': coco_eval.stats[2],
            'AP-Small': coco_eval.stats[3],
            'AP-Medium': coco_eval.stats[4],
            'AP-Large': coco_eval.stats[5],
            'AR@1': coco_eval.stats[6],    
            'AR@10': coco_eval.stats[7],   
            'AR@100': coco_eval.stats[8],  
            'AR-Small': coco_eval.stats[9],
            'AR-Medium': coco_eval.stats[10],
            'AR-Large': coco_eval.stats[11]
        })

        # 为每个类别单独计算结果（修改循环部分）
        for i, (cat_id, cat_name) in enumerate(zip(cat_ids, cat_names)):
            coco_eval.params.catIds = [cat_id]
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            results.append({
                'Category': cat_name,
                'AP@0.5:0.95': coco_eval.stats[0],
                'AP@0.5': coco_eval.stats[1],
                'AP@0.75': coco_eval.stats[2],
                'AP-Small': coco_eval.stats[3],
                'AP-Medium': coco_eval.stats[4],
                'AP-Large': coco_eval.stats[5],
                'AR@1': coco_eval.stats[6],    
                'AR@10': coco_eval.stats[7],   
                'AR@100': coco_eval.stats[8],  
                'AR-Small': coco_eval.stats[9],
                'AR-Medium': coco_eval.stats[10],
                'AR-Large': coco_eval.stats[11]
            })


        # 创建 DataFrame
        df = pd.DataFrame(results)

        # 调整列顺序（新增列排序）
        df = df[['Category', 'AP@0.5:0.95', 'AP@0.5', 'AP@0.75',
                'AP-Small', 'AP-Medium', 'AP-Large',
                'AR@1', 'AR@10', 'AR@100', 
                'AR-Small', 'AR-Medium', 'AR-Large']]


        df.to_csv(output_path, index=False, float_format='%.3f')
        # 设置打印格式（新增这三行👇）
        pd.set_option('display.float_format', '{:.3f}'.format)
        pd.set_option('display.width', 1000)  # 调整打印宽度
        pd.set_option('display.max_columns', None)  # 显示所有列
        print(f"Results saved to {output_path}")