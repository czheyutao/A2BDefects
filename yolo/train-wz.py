from ultralytics import YOLO

data = '/data/hyt/yolo/datasets/yaml/wz712.yaml'

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
            # flipud=0.8, 
            # translate=0.8, 
            # scale=0.8, 
            # mixup=0.5, 
            # copy_paste=0.5, 
            # shear=0.5, 
            # perspective=0.0005,
            fraction=0.75,
            project='yolo/yoloruns/WZ',
            name='11l-seg-75',
            device="4,5,6,7",
            seed=0
        )