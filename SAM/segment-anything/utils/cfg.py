from box import Box

config = {
    "num_devices": 4,
    "batch_size": 1,
    "num_workers": 4,
    "num_epochs": 36,
    "eval_interval": 2,
    "out_dir": "/data/hyt/segment-anything/output",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4, 
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/data/hyt/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/data/hyt/mmdetection/WZ/train",
            "annotation_file": "/data/hyt/mmdetection/WZ/annotations/instances_train.json"
        },
        "val": {
            "root_dir": "/data/hyt/mmdetection/WZ/val",
            "annotation_file": "/data/hyt/mmdetection/WZ/annotations/instances_val.json"
        }
    }
}

cfg = Box(config)