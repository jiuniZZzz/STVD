import numpy as np
import torch
from torch.utils.data import Dataset


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
