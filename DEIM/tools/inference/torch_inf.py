"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

import sys
import os
import cv2  # Added for video processing
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh] 
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

        im.save('torch_results.jpg')


def convert_to_coco_format(labels, boxes, scores, image_id):
    """
    将labels、boxes、scores转换为COCO数据格式
    """
    coco_annotations = []
    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        coco_annotations.append({
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [float(x1), float(y1), float(width), float(height)],
            "score": float(score)
        })
    return coco_annotations


def save_to_coco_json(coco_annotations, output_path):
    """
    将COCO格式的注释保存为JSON文件
    """
    with open(output_path, 'w') as f:
        json.dump({"annotations": coco_annotations}, f, indent=4)


def load_coco_dataset(json_path, images_dir):
    """
    加载COCO数据集
    :param json_path: COCO数据集的JSON文件路径
    :param images_dir: 图片存放路径
    :return: COCO数据集
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    coco_data['images_dir'] = images_dir
    return coco_data


def process_image(model, device, image_path, annotations, coco_annotations):
    """
    处理单张图像并保存结果
    :param model: 模型实例
    :param device: 设备（CPU 或 GPU）
    :param image_path: 图像路径
    :param annotations: 图像对应的标注信息
    :param coco_annotations: 存储所有图像的COCO注释列表
    """
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((1280, 1280)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(im_data, orig_size)
    labels, boxes, scores = output

    # 确保 labels, boxes, scores 是 tensor 类型
    labels = labels.cpu().numpy().tolist()
    boxes = boxes.cpu().numpy().tolist()
    scores = scores.cpu().numpy().tolist()

    # 将结果转换为COCO格式并添加到coco_annotations列表中
    coco_annotations.extend(convert_to_coco_format(labels[0], boxes[0], scores[0], image_id=annotations['image_id']))


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # 加载COCO数据集
    json_path = args.input
    images_dir = args.images_dir
    coco_data = load_coco_dataset(json_path, images_dir)

    # 创建一个空的COCO注释列表
    all_coco_annotations = []

    # 遍历每张图像并处理
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_path = os.path.join(coco_data['images_dir'], image_info['file_name'])
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        # print(annotations[0])
        # 调用process_image处理图像
        process_image(model, device, image_path, annotations[0], all_coco_annotations)

    # 保存所有图像的CO`CO注释到一个JSON文件中
    save_to_coco_json(all_coco_annotations, 'torch_results_coco.json')

    print("All images processed and results saved in a single JSON file.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('--images_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)