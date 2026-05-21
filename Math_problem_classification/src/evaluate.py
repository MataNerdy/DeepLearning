from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import DataConfig, load_math_problem_data, make_label_maps, make_train_test_split
from .dataset import MathProblemDataset
from .model import TransformerClassificationModel


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    y_true, y_pred, probas = [], [], []

    for batch in tqdm(dataloader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch)
        probs = torch.softmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        probas.extend(probs.cpu().tolist())

    return y_true, y_pred, probas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="cointegrated/rubert-tiny2")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="reports/evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DataConfig()
    df = load_math_problem_data(config)
    _, test_df = make_train_test_split(df, config)
    labels2ids, ids2labels = make_label_maps(df, config.label_column)

    model = TransformerClassificationModel(args.model_name, n_classes=len(labels2ids), freeze_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    dataset = MathProblemDataset(test_df, labels2ids, model.tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true_ids, y_pred_ids, probas = predict(model, loader, device)
    y_true = [ids2labels[i] for i in y_true_ids]
    y_pred = [ids2labels[i] for i in y_pred_ids]
    labels = [ids2labels[i] for i in sorted(ids2labels)]

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)

    error_df = test_df.copy().reset_index(drop=True)
    error_df["true_topic"] = y_true
    error_df["pred_topic"] = y_pred
    error_df["confidence"] = [max(p) for p in probas]
    error_df[error_df["true_topic"] != error_df["pred_topic"]].to_csv(output_dir / "errors.csv", index=False)


if __name__ == "__main__":
    main()
