from ultralytics import YOLO
import os

# 加载模型
model = YOLO('/data/hyt/yolo/yoloruns/WZ/11l-seg-25/weights/best.pt')
# 评估模型
model.val(
        data='/data/hyt/yolo/datasets/yaml/wz712_test.yaml',
        batch=8,
        imgsz=1280, 
        save_json=True, 
        save_conf=True,
        project='yolo/yoloruns/WZ',
        name='25',
        device="7",
    )
# 加载模型
model = YOLO('/data/hyt/yolo/yoloruns/WZ/11l-seg-50/weights/best.pt')
# 评估模型
model.val(
        data='/data/hyt/yolo/datasets/yaml/wz712_test.yaml',
        batch=8,
        imgsz=1280, 
        save_json=True, 
        save_conf=True,
        project='yolo/yoloruns/WZ',
        name='50',
        device="7",
    )
# 加载模型
model = YOLO('/data/hyt/yolo/yoloruns/WZ/11l-seg-75/weights/best.pt')
# 评估模型
model.val(
        data='/data/hyt/yolo/datasets/yaml/wz712_test.yaml',
        batch=8,
        imgsz=1280, 
        save_json=True, 
        save_conf=True,
        project='yolo/yoloruns/WZ',
        name='75',
        device="7",
    )