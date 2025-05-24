import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.weight'] = 'bold'  # 新增全局字体加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 确保坐标轴标签加粗
plt.rcParams['axes.unicode_minus'] = False

category_data = {
    "CA": {
        "train": [1012, 1577, 558],
        "val": [123, 216, 93],
        "test": [319, 514, 155],
        "sizes": ["Small", "Medium", "Large"],
        "thresholds": [1968, 7027]
    },
    "SS": {
        "train": [2185, 1987],
        "val": [351, 263],
        "test": [689, 635],
        "sizes": ["Small", "Large"],
        "thresholds": [2547]
    },
    "EG": {
        "train": [3109],
        "val": [441],
        "test": [764],
        "sizes": ["Small"]
    },
    "WS": {
        "train": [2503, 3445, 1785],
        "val": [330, 522, 248],
        "test": [731, 1113, 494],
        "sizes": ["Small", "Medium", "Large"],
        "thresholds": [1842, 7309]
    }
}

# 生成带阈值的组合标签
def generate_labels(cat_data):
    labels = []
    for cat in ["CA", "SS", "EG", "WS"]:
        data = cat_data[cat]
        sizes = data["sizes"]
        ths = data.get("thresholds", [])
        
        for i, size in enumerate(sizes):
            # 添加阈值描述
            if len(ths) == 2 and len(sizes) == 3:  # CA/WS类型
                if i == 0:
                    label = f"{cat}-{size}\n<{ths[0]}"
                elif i == 1:
                    label = f"{cat}-{size}\n{ths[0]} - {ths[1]}"
                else:
                    label = f"{cat}-{size}\n>{ths[1]}"
            elif len(ths) == 1 and len(sizes) == 2:  # SS类型
                label = f"{cat}-{size}\n<{ths[0]}" if i == 0 else f"{cat}-{size}\n>{ths[0]}"
            else:  # EG类型
                label = f"{cat}-{size}"
            labels.append(label)
    return labels

labels = generate_labels(category_data)

# 提取数据
train_values = []
val_values = []
test_values = []

for cat in ["CA", "SS", "EG", "WS"]:
    data = category_data[cat]
    for i, size in enumerate(data["sizes"]):
        train_values.append(data["train"][i])
        val_values.append(data["val"][i])
        test_values.append(data["test"][i])

# 绘图设置
x = np.arange(len(labels))
width = 0.2  # 修改柱状图宽度，从 0.28 改为 0.2
colors = ['#66b3ff', '#99ff99', '#ff9999']

plt.figure(figsize=(12, 3))  # 修改图形宽度，从 (9, 3) 扩展为 (12, 3)

# 绘制柱状图
plt.bar(x - width, train_values, width, label='Train', color=colors[0])
plt.bar(x, val_values, width, label='Validation', color=colors[1])
plt.bar(x + width, test_values, width, label='Test', color=colors[2])

# 坐标轴设置
plt.xlabel("Category-Size with Pixel Thresholds", fontsize=10)
plt.ylabel("Instance Count (log)", fontsize=10)
plt.xticks(x, labels, fontsize=10)
plt.yscale('log')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Distribution_with_Threshold_Labels.pdf', dpi=600, bbox_inches='tight', format='pdf')
plt.show()