import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体样式
plt.rcParams.update(
    {
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.grid": True,  # 启用网格
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }
)

# 示例数据 (removed Model2 data)
data = {
    "Model": ["YOLO-v11"] * 4 + ["Co-DETR"] * 4 + ["ConvNeXT V2"] * 4,
    "Category": ["CA", "SS", "EG", "WS"] * 3,
    "ar": [60.9, 39.2, 48.8, 49.3, 69.3, 44.8, 48.0, 54.1, 55.4, 32.8, 47.6, 44.8],
    "ap": [33.9, 11.5, 28.6, 21.9, 41.7, 16.7, 31.5, 26.6, 37.0, 12.5, 30.2, 22.5],
}
df = pd.DataFrame(data)

# 配置参数 (updated model names and colors)
model_colors = {
    "YOLO-v11": "#1f77b4",  # blue
    "Co-DETR": "#d62728",  # red
    "ConvNeXT V2": "#2ca02c",  # green
}

# 修改类别标记为星形，并调整标记样式
category_markers = {
    "CA": "o",  # 圆形
    "SS": "s",  # 方形
    "EG": "^",  # 上三角形
    "WS": "*",  # 星形
}

# 创建画布
plt.figure(figsize=(6, 6), dpi=600, facecolor="white")
ax = plt.gca()

plt.ylim(8, 43)  # 原范围：10-30 → 新范围：5-35
plt.xlim(25, 75)
# 绘制散点图
for model in df["Model"].unique():
    model_data = df[df["Model"] == model]
    for category in df["Category"].unique():
        subset = model_data[model_data["Category"] == category]
        plt.scatter(
            x=subset["ar"],
            y=subset["ap"],
            marker=category_markers[category],
            s=150,
            facecolors="none",
            edgecolors=model_colors[model],
            linewidths=2,
            zorder=5,
        )

# 增强网格显示
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)

# 创建自定义图例
model_legend = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=1.5,
        markersize=8,
        label=model,
    )
    for model, color in model_colors.items()
]

category_legend = [
    plt.Line2D(
        [0],
        [0],
        marker=marker,
        color="w",
        markeredgecolor="grey",
        markeredgewidth=1.5,
        markersize=8,
        label=category,

    )
    for category, marker in category_markers.items()
]

# 添加双图例
first_legend = plt.legend(
    framealpha=0.3, handles=model_legend, loc="lower right", prop={"size": 12}
)
plt.gca().add_artist(first_legend)

plt.legend(handles=category_legend, framealpha=0.3, loc="upper left", prop={"size": 12})

# 设置坐标轴标签
plt.xlabel("AR", fontsize=14)
plt.ylabel("AP", fontsize=14)


# 保存输出
plt.savefig(
    "/data/hyt/mmdetection/visual/model_performance_comparison.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=600,
)
