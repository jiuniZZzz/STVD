import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SeaFogSTVDDataset(Dataset):
    def __init__(self, npy_path, input_frames=4):
        super().__init__()
        self.data = np.load(npy_path, mmap_mode="r")
        self.input_frames = input_frames
        print(f"Dataset Loaded: {self.data.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        cond_frames = sample[: self.input_frames]
        target_frames = sample[self.input_frames :]
        cond_frames = torch.from_numpy(cond_frames).float() / 255.0
        target_frames = torch.from_numpy(target_frames).float() / 255.0
        return {"LR": cond_frames, "HR": target_frames}


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


def train_simvp(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SeaFogSTVDDataset(args.data, input_frames=args.input_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = SimVP(in_channels=args.channels, hid_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            inputs = batch["LR"].to(device)
            targets = batch["HR"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="SimVP baseline training for STVD-style dataset.")
    parser.add_argument("--data", required=True, help="Path to .npy dataset.")
    parser.add_argument("--input-frames", type=int, default=4, help="Number of input frames.")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels per frame.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Hidden channel size.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    return parser.parse_args()


if __name__ == "__main__":
    train_simvp(parse_args())
