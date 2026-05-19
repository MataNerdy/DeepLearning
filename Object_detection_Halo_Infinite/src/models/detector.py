import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator

from src.models.backbone import Backbone
from src.models.heads import ConfidenceFreeHead
from src.models.neck import FPN, PAN


class Detector(nn.Module):
    def __init__(
        self,
        backbone_model_name="efficientnet_b0",
        neck_type="pan",
        neck_n_channels=256,
        num_classes=4,
        anchor_sizes=(16, 32, 64, 101),
        anchor_ratios=(0.5, 1.0, 2.0, 3.0),
        unfreeze=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_sizes_cfg = anchor_sizes
        self.anchor_ratios_cfg = anchor_ratios
        self.anchors = None

        self.backbone = Backbone(backbone_model_name, out_indices=(-3, -2, -1), unfreeze=unfreeze)
        feature_channels = self.backbone.backbone.feature_info.channels()
        out_indices = self.backbone.backbone.feature_info.out_indices

        level_names = ["p3", "p4", "p5"]
        in_channels_dict = {
            name: feature_channels[idx]
            for name, idx in zip(level_names, out_indices)
        }

        neck_cls = PAN if neck_type.lower() == "pan" else FPN
        self.neck = neck_cls(in_channels_dict, out_channels=neck_n_channels)

        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        self.head = ConfidenceFreeHead(
            in_channels=neck_n_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    def build_anchors(self, features, input_size):
        grid_sizes = [feat.shape[-2:] for feat in features.values()]
        strides = [[input_size[0] // h, input_size[1] // w] for h, w in grid_sizes]

        anchor_generator = AnchorGenerator(
            sizes=[self.anchor_sizes_cfg] * len(grid_sizes),
            aspect_ratios=[self.anchor_ratios_cfg] * len(grid_sizes),
        )
        anchors = torch.cat(anchor_generator.grid_anchors(grid_sizes, strides), dim=0).unsqueeze(0)
        anchor_centers = (anchors[:, :, :2] + anchors[:, :, 2:]) / 2
        anchor_sizes = anchors[:, :, 2:] - anchors[:, :, :2]

        self.register_buffer("anchors", anchors)
        self.register_buffer("anchor_centers", anchor_centers)
        self.register_buffer("anchor_sizes", anchor_sizes)

    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features)
        cls_logits_list, bbox_preds_list = self.head(neck_features)

        batch_size = x.shape[0]
        all_cls, all_bbox = [], []

        for cls_logits, bbox_preds in zip(cls_logits_list, bbox_preds_list):
            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
            cls_logits = cls_logits.view(batch_size, -1, self.head.num_classes)
            all_cls.append(cls_logits)

            bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous()
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            all_bbox.append(bbox_preds)

        cls_logits = torch.cat(all_cls, dim=1)
        bbox_offsets = torch.cat(all_bbox, dim=1)

        if self.anchors is None:
            self.build_anchors(neck_features, x.shape[-2:])

        if self.training:
            return {
                "bbox_offsets": bbox_offsets,
                "cls_logits": cls_logits,
                "anchors": self.anchors,
            }

        bboxes = self.decode_bboxes(bbox_offsets)
        cls_probs = torch.softmax(cls_logits, dim=-1)
        return {
            "bboxes": bboxes,
            "cls_probs": cls_probs,
            "anchors": self.anchors,
        }

    def decode_bboxes(self, bbox_offsets):
        batch_size = bbox_offsets.shape[0]
        device = bbox_offsets.device

        anchor_centers = self.anchor_centers.to(device).expand(batch_size, -1, -1)
        anchor_sizes = self.anchor_sizes.to(device).expand(batch_size, -1, -1)

        tx = bbox_offsets[:, :, 0]
        ty = bbox_offsets[:, :, 1]
        tw = torch.clamp(bbox_offsets[:, :, 2], min=-4.0, max=2.0)
        th = torch.clamp(bbox_offsets[:, :, 3], min=-4.0, max=2.0)

        center_x = anchor_centers[:, :, 0] + torch.sigmoid(tx) * anchor_sizes[:, :, 0]
        center_y = anchor_centers[:, :, 1] + torch.sigmoid(ty) * anchor_sizes[:, :, 1]
        w = torch.exp(tw) * anchor_sizes[:, :, 0]
        h = torch.exp(th) * anchor_sizes[:, :, 1]

        return torch.stack(
            [
                center_x - w / 2,
                center_y - h / 2,
                center_x + w / 2,
                center_y + h / 2,
            ],
            dim=-1,
        )
