import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
    
import statistics

def validate(cfg, model, val_dataloader, device, epoch=0):
  model.eval() # turn the model into evaluation mode

  val_loss_list = []

  focal_loss = FocalLoss()
  dice_loss = DiceLoss()

  with torch.no_grad(): # turn off requires_grad
    for iter, data in enumerate(val_dataloader):

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
      pred_masks, iou_predictions = model(images, bboxes)
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

      loss_total = 20. * loss_focal + loss_dice + loss_iou
      val_loss_list.append(loss_total.item())

      avg_focal = loss_focal.item() / batch_size # compute average loss of a batch
      avg_dice = loss_dice.item() / batch_size
      avg_iou = loss_iou.item() / batch_size
      avg_total = loss_total.item() / batch_size

      print(f'-- Epoch: [{epoch}] Iteration: {iter+1}/{len(val_dataloader)//batch_size} --')
      print(f'Focal Loss [{loss_focal.item():.4f}] [avg: {avg_focal:.4f}]')
      print(f'Dice Loss [{loss_dice.item():.4f}] [avg: {avg_dice:.4f}]')
      print(f'IoU Loss [{loss_iou.item():8f}] [avg: {avg_iou:.8f}]')
      print(f'Total Loss [{loss_total.item():.4f}] [avg: {avg_total:.4f}] \n')
  
    total_loss_mean = statistics.mean(val_loss_list)
    print(f'Validation [{epoch}]: Total Loss: [{total_loss_mean:.4f}]')

    print(f"\nSaving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch}-valloss-{total_loss_mean:.2f}-ckpt.pth"))
