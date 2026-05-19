import numpy as np
import torch
import torchvision.transforms.functional as VF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.training.engine import filter_predictions_func


def show_predictions(model, dataset, idx=0, score_thresh=0.8, max_draw=50):
    model.eval()
    image, target = dataset[idx]
    image_tensor = image.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(image_tensor)
        preds = filter_predictions_func(outputs, score_threshold=score_thresh)[0]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(image.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(image.device)
    image_vis = (image * std + mean).clamp(0, 1)

    image_np = np.array(VF.to_pil_image(image_vis.cpu()).convert("RGB"))
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)

    for key in preds:
        preds[key] = preds[key][:max_draw]

    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if not torch.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box.tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Pred: {label} ({score:.2f})", color="red", fontsize=10)

    for box, label in zip(target["boxes"], target["labels"]):
        if not torch.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none", linestyle="dashed")
        ax.add_patch(rect)
        ax.text(x1, y2 + 5, f"GT: {label.item()}", color="green", fontsize=10)

    ax.set_title(f"Image {idx}: red = prediction, green = ground truth")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
