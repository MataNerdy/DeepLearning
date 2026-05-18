import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0


class SimpleCnn(nn.Module):
    """Five-block CNN baseline: Conv2d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = self._block(3, 8)
        self.conv2 = self._block(8, 16)
        self.conv3 = self._block(16, 32)
        self.conv4 = self._block(32, 64)
        self.conv5 = self._block(64, 96)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(96 * 5 * 5, n_classes)

    @staticmethod
    def _block(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.out(x)


def build_resnet18(n_classes: int, weights_path: str | None = None) -> nn.Module:
    """Build ResNet18 and fine-tune only layer4 and fc."""
    model = models.resnet18(weights=None)
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    for name, param in model.named_parameters():
        param.requires_grad = ("layer4" in name or "fc" in name)

    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_efficientnet_b0(n_classes: int, weights_path: str | None = None) -> nn.Module:
    """Build EfficientNet-B0 and fine-tune classifier + features.6."""
    model = efficientnet_b0(weights=None)
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name or "features.6" in name)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def build_model(model_name: str, n_classes: int, weights_path: str | None = None) -> nn.Module:
    model_name = model_name.lower()
    if model_name in {"simple", "simplecnn", "simple_cnn"}:
        return SimpleCnn(n_classes)
    if model_name in {"resnet", "resnet18"}:
        return build_resnet18(n_classes, weights_path=weights_path)
    if model_name in {"efficientnet", "efficientnet_b0", "efficientnet-b0"}:
        return build_efficientnet_b0(n_classes, weights_path=weights_path)
    raise ValueError(f"Unknown model_name={model_name}")
