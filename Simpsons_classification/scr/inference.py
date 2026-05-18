import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import DEVICE, DEFAULT_TEST_DIR
from dataset import SimpsonsDataset, load_label_encoder
from models import build_model


def predict(model, test_loader, device=DEVICE):
    model.eval()
    logits = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            logits.append(outputs)
    probs = torch.nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


def make_submission(model, test_dataset, label_encoder, output_path, batch_size=64, device=DEVICE):
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    probs = predict(model, test_loader, device=device)
    preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
    test_filenames = [path.name for path in test_dataset.files]

    submission = pd.DataFrame({"Id": test_filenames, "Expected": preds})
    submission["Id_num"] = submission["Id"].str.extract(r"img(\d+)\.jpg").astype(int)
    submission = submission.sort_values("Id_num").drop(columns="Id_num")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["simplecnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-encoder", type=Path, default=Path("outputs/label_encoder.pkl"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    args = parser.parse_args()

    label_encoder = load_label_encoder(args.label_encoder)
    test_files = sorted(args.test_dir.rglob("*.jpg"))
    test_dataset = SimpsonsDataset(test_files, mode="test", label_encoder=label_encoder)

    model = build_model(args.model, n_classes=len(label_encoder.classes_))
    state_dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    submission = make_submission(model, test_dataset, label_encoder, args.output, args.batch_size)
    print(submission.head())


if __name__ == "__main__":
    main()
