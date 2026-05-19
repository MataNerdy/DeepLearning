import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import build_loader
from losses import build_loss
from metrics import dice_score, iou_score
from models import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, dices, ious = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            dices.append(dice_score(masks, logits))
            ious.append(iou_score(masks, logits))
    return {"loss": total_loss / len(loader), "dice": sum(dices) / len(dices), "iou": sum(ious) / len(ious)}


def main():
    parser = argparse.ArgumentParser(description="Train SegNet/U-Net on PH2 lesion segmentation.")
    parser.add_argument("--model", choices=["segnet", "unet"], default="unet")
    parser.add_argument("--loss", choices=["bce", "dice", "tversky", "focal"], default="tversky")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(args.processed_dir, "train", args.batch_size, True)
    val_loader = build_loader(args.processed_dir, "val", args.batch_size, False)
    model = build_model(args.model).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice, history = -1.0, []
    ckpt_path = args.checkpoint_dir / f"{args.model}_{args.loss}_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), ckpt_path)

    with open(args.report_dir / f"history_{args.model}_{args.loss}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best checkpoint: {ckpt_path}; best val Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
