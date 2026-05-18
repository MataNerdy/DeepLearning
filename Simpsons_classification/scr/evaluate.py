import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from config import DEVICE, DEFAULT_TRAIN_DIR
from dataset import SimpsonsDataset, load_label_encoder
from models import build_model


def evaluate(model, dataloader, device=DEVICE):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--file-list", type=Path, default=None,
                        help="Optional txt with validation image paths. If absent, evaluates all data-dir images.")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["simplecnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-encoder", type=Path, default=Path("outputs/label_encoder.pkl"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    label_encoder = load_label_encoder(args.label_encoder)
    if args.file_list:
        files = [Path(line.strip()) for line in args.file_list.read_text().splitlines() if line.strip()]
    else:
        files = sorted(args.data_dir.rglob("*.jpg"))

    dataset = SimpsonsDataset(files, mode="val", label_encoder=label_encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, n_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    y_true, y_pred = evaluate(model, dataloader)
    print(f"accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()
