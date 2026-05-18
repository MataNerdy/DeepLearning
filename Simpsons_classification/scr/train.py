import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, DEFAULT_TRAIN_DIR, RANDOM_STATE
from dataset import SimpsonsDataset, save_label_encoder
from models import build_model


def fit_epoch(model, train_loader, criterion, optimizer, device=DEVICE):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        processed_data += inputs.size(0)

    return running_loss / processed_data, running_corrects / processed_data


def eval_epoch(model, val_loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            processed_data += inputs.size(0)

    return running_loss / processed_data, running_corrects / processed_data


def train_model(model, train_loader, val_loader, epochs, lr=1e-3, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = []

    model.to(device)
    for epoch in tqdm(range(epochs), desc="epoch"):
        train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"Epoch {epoch + 1:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["simplecnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Optional local pretrained weights path for Kaggle offline environment.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.train_dir.rglob("*.jpg"))
    labels = [path.parent.name for path in files]
    train_files, val_files = train_test_split(
        files,
        test_size=args.val_size,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    label_encoder = LabelEncoder().fit(labels)
    save_label_encoder(label_encoder, args.output_dir / "label_encoder.pkl")

    train_dataset = SimpsonsDataset(train_files, mode="train", label_encoder=label_encoder)
    val_dataset = SimpsonsDataset(val_files, mode="val", label_encoder=label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, n_classes=len(label_encoder.classes_), weights_path=args.weights_path)
    history = train_model(model, train_loader, val_loader, args.epochs, lr=args.lr)

    torch.save(model.state_dict(), args.output_dir / f"best_{args.model}.pth")
    with open(args.output_dir / f"history_{args.model}.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
