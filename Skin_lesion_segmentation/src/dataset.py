from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PH2SegmentationDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        self.x = np.load(x_path).astype("float32")
        self.y = np.load(y_path).astype("float32")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.x[idx]).permute(2, 0, 1)
        mask = torch.from_numpy(self.y[idx]).unsqueeze(0)
        return image, mask


def build_loader(processed_dir: Path, split: str, batch_size: int, shuffle: bool):
    dataset = PH2SegmentationDataset(processed_dir / f"x_{split}.npy", processed_dir / f"y_{split}.npy")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
