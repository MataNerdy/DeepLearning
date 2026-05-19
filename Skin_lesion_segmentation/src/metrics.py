import torch


def dice_score(targets, logits, threshold=0.5, eps=1e-8):
    preds = torch.sigmoid(logits) > threshold
    targets = targets.bool()
    intersection = torch.sum(targets & preds)
    union = torch.sum(targets) + torch.sum(preds)
    return ((2 * intersection + eps) / (union + eps)).item()


def iou_score(targets, logits, threshold=0.5, eps=1e-8):
    preds = torch.sigmoid(logits) > threshold
    targets = targets.bool()
    intersection = torch.sum(targets & preds)
    union = torch.sum(targets | preds)
    return ((intersection + eps) / (union + eps)).item()
