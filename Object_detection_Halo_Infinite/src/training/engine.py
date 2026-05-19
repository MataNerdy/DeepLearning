from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.assignment.tal import TAL_assigner
from src.losses.diou import diou_loss


def compute_loss_from_assigner(cls_logits, bbox_offsets, assign_result, class_weights=None):
    cls_targets = assign_result["cls_targets"]
    box_targets = assign_result["box_targets"]
    fg_mask = assign_result["foreground_mask"]

    batch_size, num_anchors, num_classes = cls_logits.shape
    kwargs = {}
    if class_weights is not None:
        kwargs["weight"] = class_weights.to(cls_logits.device)

    cls_loss = F.cross_entropy(
        cls_logits.view(-1, num_classes),
        cls_targets.view(-1),
        reduction="mean",
        **kwargs,
    )

    if fg_mask.any():
        bbox_loss = diou_loss(bbox_offsets[fg_mask], box_targets[fg_mask])
    else:
        bbox_loss = torch.tensor(0.0, device=cls_logits.device)

    return cls_loss, bbox_loss


def training_step(model, batch, optimizer, device, class_weights=None):
    model.train()
    images, targets = batch
    images = torch.stack([img.to(device) for img in images])

    outputs = model(images)
    bbox_offsets = outputs["bbox_offsets"]
    cls_logits = outputs["cls_logits"]
    anchors = outputs["anchors"].to(device)
    anchor_centers = ((anchors[..., :2] + anchors[..., 2:]) / 2).to(device)

    gt_boxes = [target["boxes"].to(device) for target in targets]
    gt_labels = [target["labels"].to(device) for target in targets]

    pred_boxes = model.decode_bboxes(bbox_offsets)
    assign_result = TAL_assigner(
        anchors=anchors,
        anchor_centers=anchor_centers,
        pred_cls_logits=cls_logits,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        alpha=1.0,
        beta=6.0,
        top_k=30,
    )

    cls_loss, bbox_loss = compute_loss_from_assigner(
        cls_logits,
        bbox_offsets,
        assign_result,
        class_weights=class_weights,
    )
    total_loss = 0.3 * cls_loss + bbox_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.detach().item(),
        "cls_loss": cls_loss.detach().item(),
        "bbox_loss": bbox_loss.detach().item(),
        "num_fg": assign_result["foreground_mask"].sum().item(),
    }


def filter_predictions_func(outputs, score_threshold=0.5, nms_threshold=0.3, **kwargs):
    results = []
    for boxes, scores in zip(outputs["bboxes"], outputs["cls_probs"]):
        labels = torch.argmax(scores, dim=-1) - 1
        confs = torch.max(scores, dim=-1).values
        keep = (labels >= 0) & (confs > score_threshold)

        boxes = boxes[keep]
        labels = labels[keep]
        confs = confs[keep]

        if boxes.numel() == 0:
            results.append({
                "boxes": torch.zeros((0, 4), device=boxes.device),
                "scores": torch.zeros((0,), device=boxes.device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=boxes.device),
            })
            continue

        keep_idx = torchvision.ops.nms(boxes, confs, iou_threshold=nms_threshold)
        results.append({
            "boxes": boxes[keep_idx],
            "scores": confs[keep_idx],
            "labels": labels[keep_idx],
        })
    return results


@torch.no_grad()
def validate(model, dataloader, device, score_threshold=0.1, nms_threshold=0.5):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for images, targets in tqdm(dataloader, desc="Running validation", leave=False):
        images = torch.stack(images).to(device)
        targets = [
            {k: (torch.as_tensor(v).to(device) if not torch.is_tensor(v) else v.to(device)) for k, v in target.items()}
            for target in targets
        ]

        outputs = model(images)
        predictions = filter_predictions_func(outputs, score_threshold, nms_threshold)
        metric.update(predictions, targets)

    return metric.compute()["map"].item()


def train_detector(
    model,
    dataloader,
    optimizer,
    device,
    epochs=1,
    val_dataloader=None,
    checkpoint_path="best_detector.pth",
    class_weights=None,
):
    model.to(device)
    best_map = -1.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        cls_losses, bbox_losses, num_fgs = [], [], []

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            step_out = training_step(model, batch, optimizer, device, class_weights=class_weights)
            epoch_loss += step_out["loss"]
            cls_losses.append(step_out["cls_loss"])
            bbox_losses.append(step_out["bbox_loss"])
            num_fgs.append(step_out["num_fg"])
            progress.set_postfix(loss=step_out["loss"], fg=step_out["num_fg"])

        print(
            f"[Epoch {epoch + 1}] "
            f"Total: {epoch_loss / len(dataloader):.4f} | "
            f"Cls: {np.mean(cls_losses):.4f} | "
            f"Box: {np.mean(bbox_losses):.4f} | "
            f"Avg FG: {np.mean(num_fgs):.1f}"
        )

        if val_dataloader is not None:
            map_score = validate(model, val_dataloader, device)
            print(f"[Epoch {epoch + 1}] Validation mAP: {map_score:.4f}")

            if map_score > best_map:
                best_map = map_score
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[Epoch {epoch + 1}] New best mAP: {best_map:.4f}. Saved: {checkpoint_path}")
