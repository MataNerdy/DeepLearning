import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


def dice_loss(logits, targets, eps=1e-8):
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets)
    union = torch.sum(probs) + torch.sum(targets)
    return 1 - (2 * intersection + eps) / (union + eps)


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-8):
    probs = torch.sigmoid(logits)
    tp = torch.sum(probs * targets)
    fp = torch.sum(probs * (1 - targets))
    fn = torch.sum((1 - probs) * targets)
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def build_loss(name: str):
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return dice_loss
    if name == "tversky":
        return tversky_loss
    if name == "focal":
        return lambda logits, targets: sigmoid_focal_loss(logits, targets, alpha=-1, gamma=2, reduction="mean")
    raise ValueError(f"Unknown loss: {name}")
