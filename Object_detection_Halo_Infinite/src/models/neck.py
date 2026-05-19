import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()
        self.reduce_layers = nn.ModuleDict({
            name: nn.Conv2d(in_ch, out_channels, 1)
            for name, in_ch in in_channels_dict.items()
        })
        self.top_down_blocks = nn.ModuleDict({
            "p5_p4": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p4_p3": nn.Conv2d(out_channels, out_channels, 3, padding=1),
        })
        self.output_convs = nn.ModuleDict({
            "p3": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p4": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p5": nn.Conv2d(out_channels, out_channels, 3, padding=1),
        })

    def forward(self, feats):
        p3 = self.reduce_layers["p3"](feats["p3"])
        p4 = self.reduce_layers["p4"](feats["p4"])
        p5 = self.reduce_layers["p5"](feats["p5"])

        p5_td = p5
        p4_td = self.top_down_blocks["p5_p4"](
            F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest") + p4
        )
        p3_td = self.top_down_blocks["p4_p3"](
            F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest") + p3
        )

        return {
            "p3": self.output_convs["p3"](p3_td),
            "p4": self.output_convs["p4"](p4_td),
            "p5": self.output_convs["p5"](p5_td),
        }


class PAN(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()
        self.reduce_layers = nn.ModuleDict({
            name: nn.Conv2d(in_ch, out_channels, 1)
            for name, in_ch in in_channels_dict.items()
        })
        self.top_down_blocks = nn.ModuleDict({
            "p5_p4": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p4_p3": nn.Conv2d(out_channels, out_channels, 3, padding=1),
        })
        self.bottom_up_blocks = nn.ModuleDict({
            "p3_p4": nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2),
            "p4_p5": nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2),
        })
        self.output_convs = nn.ModuleDict({
            "p3": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p4": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p5": nn.Conv2d(out_channels, out_channels, 3, padding=1),
        })

    def forward(self, feats):
        p3 = self.reduce_layers["p3"](feats["p3"])
        p4 = self.reduce_layers["p4"](feats["p4"])
        p5 = self.reduce_layers["p5"](feats["p5"])

        p5_td = p5
        p4_td = self.top_down_blocks["p5_p4"](
            F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest") + p4
        )
        p3_td = self.top_down_blocks["p4_p3"](
            F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest") + p3
        )

        p4_out = self.bottom_up_blocks["p3_p4"](p3_td) + p4_td
        p5_out = self.bottom_up_blocks["p4_p5"](p4_out) + p5_td

        return {
            "p3": self.output_convs["p3"](p3_td),
            "p4": self.output_convs["p4"](p4_out),
            "p5": self.output_convs["p5"](p5_out),
        }
