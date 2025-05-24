import matplotlib.pyplot as plt
# 增大全局字号
plt.rcParams.update({
'font.size': 14,
'font.weight': 'bold',       # 全局字体加粗
'axes.labelweight': 'bold',   # 坐标轴标签加粗
'axes.titleweight': 'bold',   # 标题加粗
})
# 训练集规模
train_sizes = [25, 50, 75, 100]

# 模型性能数据
model1_map = [10.6, 13.0, 18.7, 24.0]  
model2_map = [22.4, 26.5, 27.7, 29.1]  
model3_map = [17.0, 21.3, 23.9, 25.6]  

# 创建画布
plt.figure(figsize=(6, 3), dpi=600)

# 绘制折线
plt.plot(train_sizes, model1_map, 
         marker='o', label='YOLO-v11', 
         linestyle='-', color='#1f77b4',
         linewidth=2)

plt.plot(train_sizes, model2_map,
         marker='s', label='Co-DETR',
         linestyle='--', color='#d62728',
         linewidth=2)

plt.plot(train_sizes, model3_map,
         marker='D', label='ConvNeXT V2',
         linestyle='-.', color='#2ca02c',
         linewidth=2)

# 坐标轴设置
plt.xlabel('Training Set Size (%)')
plt.ylabel('AP')
plt.xticks(train_sizes)

# 调整y轴范围 (主要修改点)
plt.ylim(10, 30)  # 原范围：10-30 → 新范围：5-35
plt.xlim(20, 105)

# 设置自定义刻度
plt.yticks([10, 15, 20, 25, 30])  # 明确指定刻度位置

# 辅助元素
plt.legend(loc='lower right', framealpha=0.3, fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# 调整数值标注位置
vertical_offset = {
    'YOLO-v11': 0.7,
    'Co-DETR': -0.9,
    'ConvNeXT V2': 0.7
}

for x, y in zip(train_sizes[0:-1], model1_map[0:-1]):
    plt.text(x, y + vertical_offset['YOLO-v11'], f'{y:.1f}', 
             ha='center', va='bottom', fontsize=8, color='#1f77b4')
plt.text(train_sizes[-1], model1_map[-1] - 2, f'{model1_map[-1]:.1f}', 
            ha='center', va='bottom', fontsize=8, color='#1f77b4')

for x, y in zip(train_sizes, model2_map):
    plt.text(x, y + vertical_offset['Co-DETR'], f'{y:.1f}',
             ha='center', va='top', fontsize=8, color='#d62728')

for x, y in zip(train_sizes, model3_map):
    plt.text(x, y + vertical_offset['ConvNeXT V2'], f'{y:.1f}',
             ha='center', va='bottom', fontsize=8, color='#2ca02c')

plt.tight_layout()
plt.savefig('/data/hyt/mmdetection/visual/data_size_analysis.pdf', bbox_inches='tight',format='pdf')