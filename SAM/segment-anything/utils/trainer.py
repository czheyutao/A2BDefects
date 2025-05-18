import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cfg import cfg

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def train_sam(cfg, model, optimizer, scheduler, train_dataloader, val_dataloader, device):
    from tqdm import tqdm  # 添加进度条库

    train_loss = []
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    
    for epoch in range(cfg.num_epochs):
        model.train()  # 设置模型为训练模式
        epoch_train_loss = 0.0
        epoch_focal_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_iou_loss = 0.0

        for iter, data in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{cfg.num_epochs}')):  # 添加进度条
            images, bboxes, gt_masks = data

            # load batch on GPU device
            images = images.to(device)
            bboxes = torch.stack(bboxes, dim=0)
            bboxes = bboxes.cuda()
            bboxes = list(bboxes)
            gt_masks = torch.stack(gt_masks, dim=0)
            gt_masks = gt_masks.cuda()
            gt_masks = list(gt_masks)

            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes) # feed-forward
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=device)
            loss_dice = torch.tensor(0., device=device)
            loss_iou = torch.tensor(0., device=device)

            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions): 
                # compute batch_iou of pred_mask and gt_mask
                pred_mask = (pred_mask >= 0.5).float() 
                intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1,2))
                union = torch.sum(pred_mask, dim=(1,2))
                epsilon = 1e-7
                batch_iou = (intersection / (union + epsilon)).unsqueeze(1)

                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss_total.item()
            epoch_focal_loss += loss_focal.item()
            epoch_dice_loss += loss_dice.item()
            epoch_iou_loss += loss_iou.item()

        avg_focal = epoch_focal_loss / len(train_dataloader)  # compute average loss of an epoch
        avg_dice = epoch_dice_loss / len(train_dataloader)
        avg_iou = epoch_iou_loss / len(train_dataloader)
        avg_total = epoch_train_loss / len(train_dataloader)

        train_loss.append(avg_total)

        print(f'-- Epoch: [{epoch+1}] Mean Total Loss: [{avg_total:.4f}] --')
        print(f'Focal Loss [avg: {avg_focal:.4f}]')
        print(f'Dice Loss [avg: {avg_dice:.4f}]')
        print(f'IoU Loss [avg: {avg_iou:.4f}] \n')

        # 验证集验证
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():
            for val_data in tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{cfg.num_epochs}'):
                val_images, val_bboxes, val_gt_masks = val_data

                # load batch on GPU device
                val_images = val_images.to(device)
                val_bboxes = torch.stack(val_bboxes, dim=0)
                val_bboxes = val_bboxes.cuda()
                val_bboxes = list(val_bboxes)
                val_gt_masks = torch.stack(val_gt_masks, dim=0)
                val_gt_masks = val_gt_masks.cuda()
                val_gt_masks = list(val_gt_masks)

                val_batch_size = val_images.size(0)
                val_pred_masks, val_iou_predictions = model(val_images, val_bboxes)  # feed-forward
                val_num_masks = sum(len(pred_mask) for pred_mask in val_pred_masks)
                val_loss_focal = torch.tensor(0., device=device)
                val_loss_dice = torch.tensor(0., device=device)
                val_loss_iou = torch.tensor(0., device=device)

                for val_pred_mask, val_gt_mask, val_iou_prediction in zip(val_pred_masks, val_gt_masks, val_iou_predictions): 
                    # compute batch_iou of pred_mask and gt_mask
                    val_pred_mask = (val_pred_mask >= 0.5).float() 
                    val_intersection = torch.sum(torch.mul(val_pred_mask, val_gt_mask), dim=(1,2))
                    val_union = torch.sum(val_pred_mask, dim=(1,2))
                    val_epsilon = 1e-7
                    val_batch_iou = (val_intersection / (val_union + val_epsilon)).unsqueeze(1)

                    val_loss_focal += focal_loss(val_pred_mask, val_gt_mask, val_num_masks)
                    val_loss_dice += dice_loss(val_pred_mask, val_gt_mask, val_num_masks)
                    val_loss_iou += F.mse_loss(val_iou_prediction, val_batch_iou, reduction='sum') / val_num_masks

                val_loss_total = val_loss_focal + val_loss_dice + val_loss_iou
                val_loss += val_loss_total.item()

        avg_val_total = val_loss / len(val_dataloader)
        print(f'-- Validation Epoch: [{epoch+1}] Mean Total Loss: [{avg_val_total:.4f}] --\n')

def lr_lambda(step):
    if step < cfg.opt.warmup_steps:
        return step / cfg.opt.warmup_steps
    elif step < cfg.opt.steps[0]:
        return 1.0
    elif step < cfg.opt.steps[1]:
        return 1 / cfg.opt.decay_factor
    else:
        return 1 / (cfg.opt.decay_factor**2)