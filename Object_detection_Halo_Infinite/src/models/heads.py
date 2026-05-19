import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, features):
        cls_outputs, reg_outputs = [], []
        for feat in features.values():
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)
            cls_outputs.append(self.cls_head(cls_feat))
            reg_outputs.append(torch.cat([self.reg_head(reg_feat), self.obj_head(reg_feat)], dim=1))
        return cls_outputs, reg_outputs


class ConfidenceFreeHead(nn.Module):
    """Head with background as class 0 and no separate objectness branch."""

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)

        self.num_classes = num_classes + 1
        self.num_anchors = num_anchors

        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0)
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.constant_(self.reg_head.bias, 0)

    def forward(self, features):
        cls_outputs, reg_outputs = [], []
        for feat in features.values():
            x = F.relu(self.conv(feat))
            cls_outputs.append(self.cls_head(x))
            reg_outputs.append(torch.tanh(self.reg_head(x)))
        return cls_outputs, reg_outputs
