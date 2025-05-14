import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ========================= 全局配置 =========================
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})

# ========================= 数据准备 =========================
# 原始数据定义（保持与用户提供完全一致）
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

# 构建完整数据框架
model_list = []
for model in resFiles.keys():
    model_list.extend([model]*4)  # 每个模型有4个类别数据

data = {
    "Model": model_list,
    "Category": ["CA", "SS", "EG", "WS"] * len(resFiles),
    "ap": [
        # ConvNeXT V2
        37.0, 12.5, 30.2, 22.5,
        # MaskRCNN
        27.8, 8.0, 24.1, 16.2,
        # Mask2Former
        31.7, 10.7, 25.2, 19.7,
        # Co-DETR
        41.7, 16.7, 31.5, 26.6,
        # YOLO-v8
        32.8, 11.8, 27.6, 19.8,
        # YOLO-v9
        32.1, 10.7, 28.0, 19.5,
        # YOLO-v10
        34.3, 11.6, 27.7, 19.5,
        # YOLO-v11
        33.9, 11.5, 28.6, 21.9,
        # YOLOv11+SAM
        32.6, 10.5, 27.3, 20.9,
        # DEIM+SAM
        33.9, 12.1, 27.8, 21.6,
        # ConvNeXT V2+SAM
        35.1, 11.7, 26.6, 20.6,
        # Co-DETR+SAM
        39.3, 14.7, 31.5, 24.7
    ],
    "ar": [
        # ConvNeXT V2
        55.4, 32.8, 47.6, 44.8,
        # MaskRCNN
        39.5, 17.6, 37.1, 29.1,
        # Mask2Former
        49.1, 27.2, 39.0, 37.1,
        # Co-DETR
        69.3, 44.8, 48.0, 54.1,
        # YOLO-v8
        59.5, 38.4, 47.5, 48.1,
        # YOLO-v9
        61.6, 37.8, 48.4, 48.6,
        # YOLO-v10
        60.4, 37.4, 50.2, 48.2,
        # YOLO-v11
        60.9, 39.2, 48.8, 49.3,
        # YOLOv11+SAM
        50.4, 30.2, 44.4, 44.2,
        # DEIM+SAM
        56.0, 33.2, 43.4, 45.8,
        # ConvNeXT V2+SAM
        53.8, 31.9, 45.2, 43.2,
        # Co-DETR+SAM
        60.7, 35.1, 46.8, 46.6
    ]
}
df = pd.DataFrame(data)

# ========================= 可视化设置 =========================
models = list(resFiles.keys())
colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

# ========================= 生成独立图例 =========================
plt.figure(figsize=(6, 1))
legend_elements = [
    plt.Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors[idx],
               markeredgecolor=colors[idx],
               markersize=8,
               linewidth=2,
               label=model)
    for idx, model in enumerate(models)
]

ax = plt.gca()
ax.axis('off')
plt.legend(
    handles=legend_elements,
    framealpha=0.5,
    loc='center',
    ncol=3,  # 3列布局
    columnspacing=2.0,
    handletextpad=1.0,
    prop={'size': 10}
)

plt.savefig(
    "/data/hyt/mmdetection/visual/model_legend.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=600
)
plt.close()

# ========================= 生成各分类图表 =========================
categories = ["CA", "SS", "EG", "WS"]

for category in categories:
    plt.figure(figsize=(3, 3), dpi=600, facecolor="white")
    ax = plt.gca()
    
    # 坐标范围设置
    plt.ylim(df[df.Category == category].ap.min()-2, 
             df[df.Category == category].ap.max()+2)
    plt.xlim(df[df.Category == category].ar.min()-5, 
             df[df.Category == category].ar.max()+5)
    
    # 绘制所有模型点
    for idx, model in enumerate(models):
        model_data = df[(df.Model == model) & (df.Category == category)]
        plt.scatter(
            x=model_data.ar,
            y=model_data.ap,
            s=50,
            edgecolors=colors[idx],
            facecolors=colors[idx],
            linewidths=1,
            marker='o',  # 统一圆形标记
            zorder=5,
            alpha=0.9
        )
    
    # 增强可视化元素
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
    plt.xlabel("AR", fontsize=10)  # 缩小x轴标签字体
    plt.ylabel("AP", fontsize=10)  # 缩小y轴标签字体
    
    # 新增刻度设置
    ax.tick_params(axis='both', 
                  which='major',
                  labelsize=8)    # 增大刻度数字
    
    # 新增类别名称提示
    plt.text(0.05, 0.95, f"{category}", 
             transform=ax.transAxes, 
             fontsize=10, 
             fontweight='bold',
             color="grey",
             verticalalignment='top')
    
    plt.savefig(
        f"/data/hyt/mmdetection/visual/{category}_performance.pdf",
        bbox_inches="tight",
        format="pdf",
        dpi=600
    )
    plt.close()

print("可视化完成！生成文件：")
print("- 4个分类性能图：CA/SS/EG/WS_performance.pdf")
print("- 1个图例文件：model_legend.pdf")