import torch.nn as nn
import timm


class Backbone(nn.Module):
    """EfficientNet feature extractor with selected stages unfrozen."""

    def __init__(self, model_name="efficientnet_b0", out_indices=(-3, -2, -1), unfreeze=3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze > 0 and hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        return {"p3": feats[0], "p4": feats[1], "p5": feats[2]}
