# -*- coding: utf-8 -*-
"""Fine Tuning SAM.ipynb

自动从Colab生成。

原始文件位于
    https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww

## 概述
本教程由Alex Bonnet，Encord的机器学习解决方案工程师准备 - 你可能还对我们的开源框架感兴趣，该框架用于计算机视觉模型的测试、评估和验证，你可以在这里找到：https://github.com/encord-team/encord-active)。

本笔记本实现了博客文章中讨论的步骤：https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/

目标是获取Segment Anything Model并将其微调到特定应用。我们将使用stamp verification数据集 https://www.kaggle.com/datasets/rtatman/stamp-verification-staver-dataset，因为它包含SAM可能未见过的数据（发票扫描件上的邮票），具有精确的地面真实分割掩码，还有我们可以用作SAM提示的边界框。

# ## 设置
# """

# ! pip install kaggle &> /dev/null
# ! pip install torch torchvision &> /dev/null
# ! pip install opencv-python pycocotools matplotlib onnxruntime onnx &> /dev/null
# ! pip install git+https://github.com/facebookresearch/segment-anything.git &> /dev/null
# ! wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth &> /dev/null

# """**操作要求:** 将你的kaggle.json文件放入笔记本工作区的文件中。更多信息请参阅此处 https://github.com/Kaggle/kaggle-api#api-credentials"""

# ! mkdir ~/.kaggle
# ! mv kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json

# ! kaggle datasets download rtatman/stamp-verification-staver-dataset

# ! unzip stamp-verification-staver-dataset.zip &> /dev/null

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 排除具有零个或多个边界框的扫描（前100个）
stamps_to_exclude = {
    'stampDS-00008',
    'stampDS-00010',
    'stampDS-00015',
    'stampDS-00021',
    'stampDS-00027',
    'stampDS-00031',
    'stampDS-00039',
    'stampDS-00041',
    'stampDS-00049',
    'stampDS-00053',
    'stampDS-00059',
    'stampDS-00069',
    'stampDS-00073',
    'stampDS-00080',
    'stampDS-00090',
    'stampDS-00098',
    'stampDS-00100'
}.union({
    'stampDS-00012',
    'stampDS-00013',
    'stampDS-00014',
}) # 排除3个我们不希望用于微调的扫描类型

"""## 数据预处理

我们提取边界框坐标，这些坐标将被用作输入到SAM的提示。
"""

bbox_coords = {}
for f in sorted(Path('ground-truth-maps/ground-truth-maps/').iterdir())[:100]:
  k = f.stem[:-3]
  if k not in stamps_to_exclude:
    im = cv2.imread(f.as_posix())
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(contours) > 1:
      x,y,w,h = cv2.boundingRect(contours[0])
      height, width, _ = im.shape
      bbox_coords[k] = np.array([x, y, x + w, y + h])

"""我们提取地面真实分割掩码"""

ground_truth_masks = {}
for k in bbox_coords.keys():
  gt_grayscale = cv2.imread(f'ground-truth-pixel/ground-truth-pixel/{k}-px.png', cv2.IMREAD_GRAYSCALE)
  ground_truth_masks[k] = (gt_grayscale == 0)

"""## 查看图像、边界框提示和地面真实分割掩码"""

# 辅助函数由https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb提供
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

"""我们可以看到这里的地面真实掩码非常紧，这将有助于计算准确的损失。
叠加的边界框将是一个很好的提示。
"""

name = 'stampDS-00004'
image = cv2.imread(f'scans/scans/{name}.png')

plt.figure(figsize=(10,10))
plt.imshow(image)
show_box(bbox_coords[name], plt.gca())
show_mask(ground_truth_masks[name], plt.gca())
plt.axis('off')
plt.show()

"""## 准备微调"""

model_type = 'vit_b'
checkpoint = 'sam_vit_b_01ec64.pth'
device = 'cuda:0'

from segment_anything import SamPredictor, sam_model_registry
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train();

"""我们转换输入图像为SAM内部函数期望的格式。"""

# 预处理图像
from collections import defaultdict

import torch

from segment_anything.utils.transforms import ResizeLongestSide

transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
  image = cv2.imread(f'scans/scans/{k}.png')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  transform = ResizeLongestSide(sam_model.image_encoder.img_size)
  input_image = transform.apply_image(image)
  input_image_torch = torch.as_tensor(input_image, device=device)
  transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

  input_image = sam_model.preprocess(transformed_image)
  original_image_size = image.shape[:2]
  input_size = tuple(transformed_image.shape[-2:])

  transformed_data[k]['image'] = input_image
  transformed_data[k]['input_size'] = input_size
  transformed_data[k]['original_image_size'] = original_image_size

# 设置优化器，超参数调整将提高性能
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(bbox_coords.keys())

"""## 运行微调

这是主要的训练循环。

改进包括批量处理和将图像和提示嵌入的计算移出循环，因为我们不调整模型的这些部分，这将加快训练速度，因为我们在每个epoch中不应重新计算嵌入。有时优化器在参数空间中迷失方向，损失函数会爆炸。从头开始重新运行（包括运行'准备微调'以下的所有单元格，以使用默认权重重新开始）将解决这个问题。

在生产实现中，优化器/损失函数的更好选择肯定会有所帮助。
"""

from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

num_epochs = 100
losses = []

for epoch in range(num_epochs):
  epoch_losses = []
  # 只在前20个示例上训练
  for k in keys[:20]:
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']

    # 这里不需要梯度，因为我们不想优化编码器
    with torch.no_grad():
      image_embedding = sam_model.image_encoder(input_image)

      prompt_box = bbox_coords[k]
      box = transform.apply_boxes(prompt_box, original_image_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
      box_torch = box_torch[None, :]

      sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
          points=None,
          boxes=box_torch,
          masks=None,
      )
    low_res_masks, iou_predictions = sam_model.mask_decoder(
      image_embeddings=image_embedding,
      image_pe=sam_model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=False,
    )

    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

    loss = loss_fn(binary_mask, gt_binary_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_losses.append(loss.item())
  losses.append(epoch_losses)
  print(f'EPOCH: {epoch}')
  print(f'平均损失: {mean(epoch_losses)}')

mean_losses = [mean(x) for x in losses]
mean_losses

plt.plot(list(range(len(mean_losses))), mean_losses)
plt.title('平均epoch损失')
plt.xlabel('Epoch编号')
plt.ylabel('损失')

plt.show()

"""## 我们可以将我们的微调模型与原始模型进行比较"""

# 使用默认权重加载模型
sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device);

# 为微调和原始模型设置预测器
from segment_anything import sam_model_registry, SamPredictor
predictor_tuned = SamPredictor(sam_model)
predictor_original = SamPredictor(sam_model_orig)

# 模型没有看到keys[21]（或keys[20]），因为我们只在keys[:20]上进行了训练
k = keys[21]
image = cv2.imread(f'scans/scans/{k}.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor_tuned.set_image(image)
predictor_original.set_image(image)

input_bbox = np.array(bbox_coords[k])

masks_tuned, _, _ = predictor_tuned.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

masks_orig, _, _ = predictor_original.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

"""我们可以看到这里的微调模型已经开始忽略单词之间的空白，这正是地面真实显示的。通过进一步训练、更多数据和进一步的超参数调整，我们将能够改进这个结果。

如果图像由于大小限制无法渲染，你可以在https://drive.google.com/file/d/1ip5aryaxcp8JcEaZubL76oOM6srVzKsc/view?usp=sharing 查看它。
"""

# 注释掉的IPython魔术以确保Python兼容性。
# %matplotlib inline
_, axs = plt.subplots(1, 2, figsize=(25, 25))


axs[0].imshow(image)
show_mask(masks_tuned, axs[0])
show_box(input_bbox, axs[0])
axs[0].set_title('使用微调模型的掩码', fontsize=26)
axs[0].axis('off')


axs[1].imshow(image)
show_mask(masks_orig, axs[1])
show_box(input_bbox, axs[1])
axs[1].set_title('使用原始模型的掩码', fontsize=26)
axs[1].axis('off')

plt.show()