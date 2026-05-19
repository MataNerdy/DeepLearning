import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from src.data.dataset import HaloDataset, collate_fn
from src.data.load_data import load_halo_split
from src.data.transforms import get_train_transform, get_val_transform
from src.models.detector import Detector
from src.training.engine import train_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="best_detector.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    train_df = load_halo_split("train")
    val_df = load_halo_split("validation")

    train_dataset = HaloDataset(train_df, transform=get_train_transform())
    val_dataset = HaloDataset(val_df, transform=get_val_transform())

    label_counter = Counter()
    for _, target in train_dataset:
        label_counter.update(target["labels"].tolist())

    num_classes = 4
    class_counts = torch.tensor([label_counter.get(i, 1) for i in range(num_classes)], dtype=torch.float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    class_weights = torch.cat([torch.tensor([0.05]), weights])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Detector(unfreeze=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_detector(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=args.device,
        epochs=args.epochs,
        val_dataloader=val_loader,
        checkpoint_path=args.checkpoint,
        class_weights=class_weights,
    )


if __name__ == "__main__":
    main()
