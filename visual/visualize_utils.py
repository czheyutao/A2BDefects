import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_points(coords, labels, ax, marker_size=375):
    """
    在指定的matplotlib坐标系中绘制点
    参数:
    coords (np.ndarray): 二维数组，表示点的坐标
    labels (np.ndarray): 一维数组，表示点的标签
    ax (matplotlib.axes.Axes): 用于绘制图形的matplotlib坐标系对象
    marker_size (int, optional): 点的大小，默认为375
    返回值:
    None
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

# 新增IoU计算函数
def box_iou(box1, box2):
    """
    计算两个边界框的IoU
    参数格式均为[x0, y0, x1, y1]
    """
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# 新增NMS处理函数
def nms(boxes, scores, labels, iou_threshold=0.5):
    """
    对同一类别的检测框执行非极大值抑制
    返回保留的框索引列表
    """
    keep_indices = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        
        # 修改排序依据：按边界框面积从大到小排序（原先是按分数排序）
        areas = (class_boxes[:,2] - class_boxes[:,0]) * (class_boxes[:,3] - class_boxes[:,1])
        sorted_idx = np.argsort(areas)[::-1]  # 改为按面积降序排列
        sorted_boxes = class_boxes[sorted_idx]
        sorted_indices = class_indices[sorted_idx]
        
        while len(sorted_boxes) > 0:
            keep_idx = sorted_indices[0]
            keep_indices.append(keep_idx)
            
            if len(sorted_boxes) == 1:
                break
                
            current_box = sorted_boxes[0]
            other_boxes = sorted_boxes[1:]
            ious = np.array([box_iou(current_box, b) for b in other_boxes])
            
            keep_mask = ious < iou_threshold
            sorted_boxes = other_boxes[keep_mask]
            sorted_indices = sorted_indices[1:][keep_mask]
    
    return keep_indices

def visualize_results(
    image: np.ndarray,
    masks: list[np.ndarray],
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    output_path: str,
    axis_off: bool = True,
    dpi: int = 600,
    show_box_label: bool = True,
) -> None:
    """
    可视化SAM预测结果并保存，输出图像尺寸与原始图像一致

    参数:
        image: 原始图像数组 (H,W,3)
        masks: 预测的mask数组列表，每个元素为(H,W)的bool数组
        boxes: 对应的bbox坐标列表，每个元素为[x1,y1,x2,y2]
        output_path: 结果保存路径
        random_color: 是否使用随机颜色，默认True
        axis_off: 是否隐藏坐标轴，默认True
        dpi: 输出图像的分辨率（每英寸点数），默认300。调整此参数可影响图像清晰度，但输出尺寸固定为原图像素尺寸。
    """
    def show_mask(mask, ax, color):
        """
        在指定的matplotlib坐标系中绘制mask

        参数:
        mask (np.ndarray): 二维布尔数组，表示要绘制的mask
        ax (matplotlib.axes.Axes): 用于绘制图形的matplotlib坐标系对象
        random_color (bool, optional): 是否使用随机颜色，默认为False

        返回值:
        None
        """
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image, interpolation='nearest')  # 新增插值参数保持锐利边缘

    def show_box(box, ax, color, label, class_id):  # 新增class_id参数
        """
        在指定的matplotlib坐标系中绘制矩形边界框
        参数:
        box (list/tuple): 包含四个数值的边界框坐标，格式为[x0, y0, x1, y1]
        ax (matplotlib.axes.Axes): matplotlib坐标系对象
        color (np.ndarray): RGBA颜色数组，用于边框颜色
        class_id (int): 类别编号（1-4）用于确定标签位置
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h,
                edgecolor=tuple(color),  # 使用传入的颜色
                facecolor=(0, 0, 0, 0),
                lw=1
            )
        )
        

    def add_label_to_box(box, ax, color,label, class_id):
        """
        在边界框上添加标签
        参数:
        ax (matplotlib.axes.Axes): matplotlib坐标系对象
        box (list/tuple): 边界框坐标 [x0, y0, x1, y1]
        label (str): 标签文本
        color (np.ndarray): RGBA颜色数组
        class_id (int): 类别编号（1-4）用于确定标签位置
        """
        # 根据class_id设置标签位置和对齐方式
        if class_id == 1:  # 左上角
            x, y = box[0], box[1]
            ha, va = 'left', 'top'
        elif class_id == 2:  # 右上角
            x, y = box[2], box[1]
            ha, va = 'right', 'top'
        elif class_id == 3:  # 右下角
            x, y = box[2], box[3]
            ha, va = 'right', 'bottom'
        else:  # 左下角（class_id ==4）
            x, y = box[0], box[3]
            ha, va = 'left', 'bottom'

        ax.text(
            x=x,
            y=y,
            s=label,
            color='white',
            fontsize=5,
            horizontalalignment=ha,  # 添加水平对齐参数
            verticalalignment=va,    # 添加垂直对齐参数
            bbox=dict(
                facecolor=tuple(color),  # 修正为使用传入的color参数
                alpha=0.7,
                edgecolor='none',
                boxstyle='round,pad=0.2'
            )
        )

    # 修正图像尺寸计算逻辑
    H, W = image.shape[:2]
    figsize = (W / dpi*4, H / dpi*4)  # 删除原*2的放大系数，保持原始比例
    plt.figure(figsize=figsize)
    plt.imshow(image)

    # 执行NMS预处理
    if len(boxes) > 0:
        boxes_np = np.array(boxes)
        scores_np = np.array(scores)
        labels_np = np.array(labels)
        
        keep_indices = nms(boxes_np, scores_np, labels_np, 0.5)
        
        # 过滤保留的检测项
        masks = [masks[i] for i in keep_indices]
        boxes = [boxes[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]
    
    # 获取颜色，共4种颜色, 存放在colors中
    colors = [
        np.array([30 / 255, 144 / 255, 255 / 255, 0.3]),  # 蓝色
        np.array([255 / 255, 50 / 255, 50 / 255, 0.3]),    # 亮红色
        np.array([255 / 255, 255 / 255, 51 / 255, 0.3]),   # 明黄色
        np.array([50 / 255, 205 / 255, 50 / 255, 0.3]),    # 翠绿色
    ]
    # colors = [
    #     np.array([255 / 255, 50 / 255, 50 / 255, 0.7]),    # 亮红色
    #     np.array([255 / 255, 255 / 255, 51 / 255, 0.7]),   # 明黄色
    #     np.array([50 / 255, 205 / 255, 50 / 255, 0.7]),    # 翠绿色
    #     np.array([30 / 255, 144 / 255, 255 / 255, 0.7]), # 蓝色
    # ]

    # 批量绘制mask和box
    for mask, box, score, label in zip(masks, boxes, scores, labels):
        # 只画置信度大于0.2的mask
        if score < 0.2:
            continue

        # if label == 1:
        #     continue

        # 同种label使用同种颜色
        class_id = label      
        mask_color = colors[class_id-1]
        border_color = mask_color.copy()
        border_color[3] = 1.0
        labels = ["CA","SS","EG","WS"] 
        # labels= ["crack-brick","brick","broken_brick","crack"]
        label = labels[class_id-1]
        
        show_mask(mask, plt.gca(), mask_color)
        # if show_box_label:
        #     show_box(box, plt.gca(), border_color, label, class_id)  # 新增class_id参数
        #     add_label_to_box(box, plt.gca(), border_color, label, class_id)

    if axis_off:
        plt.axis("off")

    # 同时保存PNG和PDF格式
    plt.savefig(output_path+'.png', dpi=dpi, bbox_inches='tight', pad_inches=0, format='png')
    plt.savefig(output_path+'.pdf', dpi=dpi, bbox_inches='tight', pad_inches=0, format='pdf')  # 启用矢量格式保存
    plt.close()
