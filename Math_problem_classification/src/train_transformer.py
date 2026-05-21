from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import DataConfig, load_math_problem_data, make_label_maps, make_train_test_split
from .dataset import MathProblemDataset
from .model import TransformerClassificationModel
from .utils import save_json, set_seed


def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu", use_amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and str(device).startswith("cuda"))

    losses = []
    y_true, y_pred = [], []

    for batch in tqdm(dataloader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train), torch.amp.autocast(
            "cuda", enabled=use_amp and str(device).startswith("cuda")
        ):
            logits = model(**batch)
            loss = criterion(logits, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def train_transformer(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = DataConfig(test_size=args.test_size, random_state=args.seed)
    df = load_math_problem_data(data_config)
    train_df, test_df = make_train_test_split(df, data_config)
    labels2ids, ids2labels = make_label_maps(df, data_config.label_column)

    model = TransformerClassificationModel(
        base_transformer_model=args.model_name,
        n_classes=len(labels2ids),
        dropout=args.dropout,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    train_dataset = MathProblemDataset(train_df, labels2ids, model.tokenizer, max_length=args.max_length)
    test_dataset = MathProblemDataset(test_df, labels2ids, model.tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(labels2ids, output_dir / "labels2ids.json")
    save_json(ids2labels, output_dir / "ids2labels.json")

    best_acc = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.use_amp)
        val_metrics = run_epoch(model, test_loader, criterion, None, device, args.use_amp)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(row)

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            bad_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best val accuracy: {best_acc:.4f}")
            break

    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="cointegrated/rubert-tiny2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints/rubert_tiny2_unfrozen")
    return parser


def main():
    args = build_parser().parse_args()
    train_transformer(args)


if __name__ == "__main__":
    main()
