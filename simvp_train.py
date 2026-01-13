import argparse
import torch
from torch.utils.data import DataLoader

from data.seafog_dataset import SeaFogSTVDDataset
from model.simvp import SimVP
from trainer_utils import train_forecast_model


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
    parser.add_argument(
        "--save-path",
        default="simvp_checkpoint.pt",
        help="Path to save the trained model checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = SeaFogSTVDDataset(args.data, input_frames=args.input_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = SimVP(in_channels=args.channels, hid_channels=args.hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_forecast_model(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        optimizer=optimizer,
        save_path=args.save_path,
    )
