import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc0 = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.out = nn.Conv2d(8, out_channels, 1)

    def forward(self, x):
        sizes, indices = [], []
        for block in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            x = block(x)
            sizes.append(x.size())
            x, idx = self.pool(x)
            indices.append(idx)
        x = self.bottleneck(x)
        for block, idx, size in zip([self.dec4, self.dec3, self.dec2, self.dec1], reversed(indices[1:]), reversed(sizes[1:])):
            x = self.unpool(x, idx, output_size=size)
            x = block(x)
        x = self.unpool(x, indices[0], output_size=sizes[0])
        return self.out(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc0 = self.block(in_channels, 8)
        self.enc1 = self.block(8, 8)
        self.enc2 = self.block(8, 16)
        self.enc3 = self.block(16, 32)
        self.enc4 = self.block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.block(64, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec4 = self.block(128, 32)
        self.up3 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec3 = self.block(64, 16)
        self.up2 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec2 = self.block(32, 8)
        self.up1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.dec1 = self.block(16, 8)
        self.up0 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.dec0 = self.block(16, 8)
        self.out = nn.Conv2d(8, out_channels, 1)

    @staticmethod
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
        )

    @staticmethod
    def crop_to(x, target):
        return x[:, :, : target.shape[2], : target.shape[3]]

    def forward(self, x):
        x0 = self.enc0(x)
        e0 = self.pool(x0)
        x1 = self.enc1(e0)
        e1 = self.pool(x1)
        x2 = self.enc2(e1)
        e2 = self.pool(x2)
        x3 = self.enc3(e2)
        e3 = self.pool(x3)
        x4 = self.enc4(e3)
        e4 = self.pool(x4)
        b = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([self.crop_to(self.up4(b), x4), x4], dim=1))
        d3 = self.dec3(torch.cat([self.crop_to(self.up3(d4), x3), x3], dim=1))
        d2 = self.dec2(torch.cat([self.crop_to(self.up2(d3), x2), x2], dim=1))
        d1 = self.dec1(torch.cat([self.crop_to(self.up1(d2), x1), x1], dim=1))
        d0 = self.dec0(torch.cat([self.crop_to(self.up0(d1), x0), x0], dim=1))
        return self.out(d0)


def build_model(name: str):
    name = name.lower()
    if name == "segnet":
        return SegNet()
    if name in {"unet", "u-net"}:
        return UNet()
    raise ValueError(f"Unknown model: {name}")
