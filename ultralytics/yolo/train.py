from ultralytics import YOLO

data = './datasets/yaml/wz712.yaml'

# 加载模型结构
model = YOLO('yolo11l-seg.yaml')

# # 读取预训练模型权重
model = model.load("yolo11l-seg.pt")  # build from YAML and transfer weigh

# 训练模型
model.train(task='segment', 
            data=data,
            epochs=400,
            patience=400,
            imgsz=1280,
            batch=16, 
            deterministic=True, 
            conf=0.000001, 
            # fraction=0.75,
            project='yolo/yoloruns/WZ',
            name='11l',
            device="4,5,6,7",
            seed=0
        )