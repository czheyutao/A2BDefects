import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def plot_pr_curve_all_categories(annFile, resFiles, iou_thr=0.5, area_rng='all', max_dets=100):
    """
    参数说明：
    annFile: COCO标注文件路径
    resFiles: 模型名称到检测结果路径的字典映射
    iou_thr: IoU阈值（默认0.5）
    area_rng: 面积范围筛选（'all'/'small'/'medium'/'large'）
    max_dets: 每张图片最大检测数量
    """
    # 加载标注数据
    cocoGt = COCO(annFile)
    
    # 获取所有类别ID
    cat_ids = cocoGt.getCatIds()
    
    plt.figure(figsize=(6, 3),dpi=600)
    
    # 面积范围映射
    area_map = {
        'all': [0**2, 1e5**2],
        'small': [0**2, 32**2],
        'medium': [32**2, 96**2],
        'large': [96**2, 1e5**2]
    }

    # 使用 plt.cm.tab20 颜色映射，确保颜色不重复
    colors = plt.cm.tab20(np.linspace(0, 1, len(resFiles)))

    for idx, (model_name, resFile) in enumerate(resFiles.items()):
        # 加载检测结果
        cocoDt = cocoGt.loadRes(resFile)
        
        # 初始化评估器
        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        
        # 配置评估参数
        cocoEval.params.iouThrs = [iou_thr]       # 使用指定IoU阈值
        cocoEval.params.catIds = cat_ids          # 所有类别
        cocoEval.params.areaRng = [area_map[area_rng]]
        cocoEval.params.maxDets = [max_dets]
        
        # 执行评估
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        # 提取precision数据 [T, R, K, A, M]
        precision = cocoEval.eval['precision']
        
        # 计算所有类别的平均精度（维度说明见下方）
        # T=0（仅使用第一个IoU阈值）
        # K=0: 所有类别取平均
        # A=0（第一个面积范围）
        # M=0（第一个最大检测数设置）
        avg_precision = precision[0, :, :, 0, 0].mean(axis=1)  # 在类别维度取平均
        
        # 生成召回率坐标轴（标准101点）
        recall = np.linspace(0, 1, 101)
        
        # 绘制曲线，使用颜色映射分配颜色
        plt.plot(recall, avg_precision, lw=1.5, label=f'{model_name}', color=colors[idx])

    # plt.title(f'PR Curve (IoU={iou_thr}, All Categories, Area={area_rng})')
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.grid(True, alpha=0.3)
    # plt.legend(loc='lower left', fontsize=10, framealpha=0.3)
    # 去掉边缘
    plt.tight_layout()
    plt.savefig(f'/data/hyt/mmdetection/visual/pr_curve_no_legend_{area_rng}.pdf', format='pdf')
    plt.show()

# 使用示例
if __name__ == '__main__':
    annFile = '/data/hyt/mmdetection/WZ/annotations/instances_test.json'
    
    # 不同模型的检测结果
    resFiles = {
        'ConvNeXT V2': '/data/hyt/mmdetection/convnext100/coco_instance/test.segm.json',
        'MaskRCNN': '/data/hyt/mmdetection/maskrcnn/coco_instance/test.segm.json',
        'Mask2Former': '/data/hyt/mmdetection/mask2former/coco_instance/test.segm.json',
        'Co-DETR': '/data/hyt/Co-DETR-main/condino.segm100.json',
        'YOLO-v8': '/data/hyt/yolo/yoloruns/WZ/test-v8l-seg-no/predictions.json',
        'YOLO-v9': '/data/hyt/yolo/yoloruns/WZ/test-v9c-seg-no/predictions.json',
        'YOLO-v10': '/data/hyt/yolo/yoloruns/WZ/test-v10l-seg-no/predictions.json',
        'YOLO-v11': '/data/hyt/yolo/yoloruns/WZ/test-11l-seg-no/predictions.json',
        'YOLOv11+SAM': '/data/hyt/SAM/results/test_yolo11_sam.json',
        'DEIM+SAM': '/data/hyt/SAM/results/test_deim_sam.json',
        'ConvNeXT V2+SAM': '/data/hyt/SAM/results/test_convnext_sam.json',
        'Co-DETR+SAM': '/data/hyt/SAM/results/test_codino_sam.json'
    }
    

    # 增大全局字号
    plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',       # 全局字体加粗
    'axes.labelweight': 'bold',   # 坐标轴标签加粗
    'axes.titleweight': 'bold',   # 标题加粗
    })
    # 绘制曲线
    plot_pr_curve_all_categories(
        annFile=annFile,
        resFiles=resFiles,
        iou_thr=0.5,
        area_rng='all'
    )