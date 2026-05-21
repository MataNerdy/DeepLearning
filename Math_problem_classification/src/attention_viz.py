from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


@torch.no_grad()
def plot_first_layer_attention(
    model,
    tokenizer,
    text: str,
    max_length: int = 128,
    output_path: str | Path | None = None,
):
    """Plot all attention heads from the first transformer layer for one text."""
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    logits, attentions = model(**encoded, output_attentions=True)
    first_layer = attentions[0][0].detach().cpu()

    n_heads = first_layer.shape[0]
    n_cols = min(4, n_heads)
    n_rows = math.ceil(n_heads / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        ax.imshow(first_layer[head_idx].numpy())
        ax.set_title(f"Head {head_idx}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for ax in axes[n_heads:]:
        ax.axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig
