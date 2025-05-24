from ultralytics import YOLO
import os

# 加载模型
model = YOLO('./yoloruns/WZ/11l/weights/best.pt')
# 评估模型
model.val(
        data='./yaml/wz712_test.yaml',
        batch=8,
        imgsz=1280, 
        save_json=True, 
        save_conf=True,
        project='yolo/yoloruns/WZ',
        name='val',
        device="4,5,6,7",
    )