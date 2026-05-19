import argparse
from pathlib import Path

import torch

from dataset import build_loader
from losses import build_loss
from metrics import dice_score, iou_score
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--model", choices=["segnet", "unet"], default="unet")
    parser.add_argument("--loss", choices=["bce", "dice", "tversky", "focal"], default="tversky")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or Path("checkpoints") / f"{args.model}_{args.loss}_best.pt"
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    criterion = build_loss(args.loss)
    loader = build_loader(args.processed_dir, "test", args.batch_size, False)

    losses, dices, ious = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            losses.append(float(criterion(logits, masks).cpu()))
            dices.append(dice_score(masks, logits))
            ious.append(iou_score(masks, logits))
    print({"test_loss": sum(losses)/len(losses), "test_dice": sum(dices)/len(dices), "test_iou": sum(ious)/len(ious)})


if __name__ == "__main__":
    main()
