from __future__ import annotations

import argparse

import torch

from .data import DataConfig, load_math_problem_data, make_label_maps
from .model import TransformerClassificationModel


@torch.no_grad()
def predict_single_problem(model, tokenizer, text: str, ids2labels: dict[int, str], top_k: int = 3, max_length: int = 256):
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    logits = model(**encoded)
    probs = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()))

    return [
        {"topic": ids2labels[int(idx)], "probability": float(prob)}
        for prob, idx in zip(values.cpu(), indices.cpu())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="cointegrated/rubert-tiny2")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_math_problem_data(DataConfig())
    labels2ids, ids2labels = make_label_maps(df)

    model = TransformerClassificationModel(args.model_name, n_classes=len(labels2ids), freeze_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    predictions = predict_single_problem(model, model.tokenizer, args.text, ids2labels, args.top_k)
    for row in predictions:
        print(f"{row['topic']}: {row['probability']:.4f}")


if __name__ == "__main__":
    main()
