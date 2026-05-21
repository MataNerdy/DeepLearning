import matplotlib.pyplot as plt
import numpy as np


def plot_history(losses, accuracies, save_path=None):
    plt.figure(figsize=(10, 4))

    plt.plot(losses, label="train loss")
    plt.plot(accuracies, label="validation accuracy")

    plt.xlabel("Epoch")
    plt.title("Training history")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def summarize_results(results):
    rows = []

    for name, metrics in results.items():
        rows.append({
            "model": name,
            "best_acc": max(metrics["accuracy"]),
            "best_epoch": int(np.argmax(metrics["accuracy"])),
            "min_loss": min(metrics["loss"]),
        })

    return rows