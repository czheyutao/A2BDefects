import torch

def convert_ckpt_to_pth(ckpt_path, pth_path):
    # 加载 .ckpt 文件
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 提取模型权重
    model_state_dict = ckpt['state_dict']
    new_state_dict = {}

    # 遍历原始字典中的键值对
    for old_key, old_value in model_state_dict.items():
        # 删除键中的 'net.' 部分
        new_key = old_key.replace('model.', '')
    
        # 使用新的键来保存值
        new_state_dict[new_key] = old_value


    # print(f"Model state dict keys: {new_state_dict.keys()}")  # 打印模型权重的键
    # 保存为 .pth 文件
    torch.save(new_state_dict, pth_path)
    print(f"Successfully converted {ckpt_path} to {pth_path}")

if __name__ == "__main__":
    # 定义输入和输出路径
    ckpt_path = "./wz_b/step=2500-val_per_mask_iou=0.81.ckpt"  # 替换为您的 .ckpt 文件路径
    pth_path = "./wz_b/step=2500-val_per_mask_iou=0.81.pth"    # 替换为您希望保存的 .pth 文件路径
    
    # 调用转换函数
    convert_ckpt_to_pth(ckpt_path, pth_path)