import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import build_loader
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Save validation/test prediction examples.")
    parser.add_argument("--model", choices=["segnet", "unet"], default="unet")
    parser.add_argument("--loss", default="tversky")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.png"))
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or Path("checkpoints") / f"{args.model}_{args.loss}_best.pt"
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    loader = build_loader(args.processed_dir, args.split, batch_size=args.n, shuffle=False)
    images, masks = next(iter(loader))
    with torch.no_grad():
        preds = torch.sigmoid(model(images.to(device))).cpu() > 0.5

    fig, axes = plt.subplots(args.n, 3, figsize=(9, 3 * args.n))
    for i in range(args.n):
        axes[i, 0].imshow(images[i].permute(1, 2, 0))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(preds[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
