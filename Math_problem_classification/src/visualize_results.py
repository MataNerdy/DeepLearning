from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_model_comparison(metrics_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(metrics_csv).sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["model"], df["macro_f1"])
    ax.set_xlabel("Macro F1")
    ax.set_title("Model comparison by Macro F1")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)


def plot_class_distribution(df: pd.DataFrame, label_column: str, output_path: str | Path) -> None:
    counts = df[label_column].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
