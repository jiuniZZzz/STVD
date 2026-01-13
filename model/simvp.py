import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimVP(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hid_channels, stride=2),
            ConvBlock(hid_channels, hid_channels, stride=1),
            ConvBlock(hid_channels, hid_channels, stride=2),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(hid_channels, hid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hid_channels, hid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(hid_channels, hid_channels, stride=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hid_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        batch, frames, channels, height, width = x.shape
        x = x.reshape(batch * frames, channels, height, width)
        x = self.encoder(x)
        _, hid_channels, h_down, w_down = x.shape
        x = x.reshape(batch, frames, hid_channels, h_down, w_down).permute(0, 2, 1, 3, 4)
        x = self.temporal(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, hid_channels, h_down, w_down)
        x = self.decoder(x)
        return x.reshape(batch, frames, channels, height, width)
