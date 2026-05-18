from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from config import DEVICE


def imshow(inp, title=None, plt_ax=plt):
    """Show normalized torch tensor as image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def plot_history(histories: dict, output_path: str | Path | None = None):
    """Plot validation accuracy and validation loss for several models."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        val_acc = [epoch["val_acc"] if isinstance(epoch, dict) else epoch[3] for epoch in history]
        plt.plot(val_acc, label=name)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        val_loss = [epoch["val_loss"] if isinstance(epoch, dict) else epoch[2] for epoch in history]
        plt.plot(val_loss, label=name)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def predict_one_sample(model, inputs, device=DEVICE):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        logits = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
    return probs


def plot_prediction_grid(model, dataset, label_encoder, nrows=3, ncols=3, output_path=None, device=DEVICE):
    """Plot random validation samples with predicted label and confidence."""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12), sharey=True, sharex=True)

    for fig_x in ax.flatten():
        index = random.randint(0, len(dataset) - 1)
        image, label = dataset[index]
        true_label = " ".join(label_encoder.inverse_transform([label])[0].split("_")).title()

        imshow(image.data.cpu(), title=true_label, plt_ax=fig_x)
        fig_x.add_patch(patches.Rectangle((0, 53), 86, 35, color="white"))

        probs = predict_one_sample(model, image.unsqueeze(0), device=device)
        predicted_proba = np.max(probs) * 100
        predicted_class_idx = np.argmax(probs)
        predicted_label = label_encoder.classes_[predicted_class_idx]
        predicted_label = predicted_label[:len(predicted_label)//2] + "\n" + predicted_label[len(predicted_label)//2:]
        fig_x.text(1, 59, f"{predicted_label}: {predicted_proba:.0f}%",
                   horizontalalignment="left", verticalalignment="top",
                   fontsize=8, color="black", fontweight="bold")

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
