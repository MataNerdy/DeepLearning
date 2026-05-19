import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    """Convert [x, y, width, height] bbox to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def normalize_boxes(boxes_raw: Any) -> List[List[float]]:
    """Normalize parquet bbox field to a list of COCO-format boxes."""
    if isinstance(boxes_raw, np.ndarray):
        boxes_raw = boxes_raw.tolist()

    if isinstance(boxes_raw, list) and len(boxes_raw) > 0:
        if isinstance(boxes_raw[0], (list, tuple, np.ndarray)):
            return [list(map(float, b)) for b in boxes_raw]
        if len(boxes_raw) == 4 and isinstance(boxes_raw[0], (int, float, np.integer, np.floating)):
            return [list(map(float, boxes_raw))]

    return []


def normalize_labels(labels_raw: Any) -> List[int]:
    """Normalize category field to zero-based class ids."""
    if isinstance(labels_raw, np.ndarray):
        labels_raw = labels_raw.tolist()
    if isinstance(labels_raw, (int, np.integer)):
        labels = [int(labels_raw)]
    else:
        labels = [int(x) for x in labels_raw]
    return [label - 1 for label in labels]


class HaloDataset(Dataset):
    """Dataset wrapper for the Halo Infinite object detection parquet files."""

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Any] = None):
        df_objects = pd.json_normalize(dataframe["objects"])[["bbox", "category"]]
        df_images = pd.json_normalize(dataframe["image"])[["bytes"]]
        self.data = dataframe[["image_id"]].join(df_objects).join(df_images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.data.iloc[idx]
        image = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
        image_np = np.array(image)

        boxes_coco = normalize_boxes(row["bbox"])
        labels = normalize_labels(row["category"])

        if self.transform is not None:
            transformed = self.transform(image=image_np, bboxes=boxes_coco, labels=labels)
            image_tensor = transformed["image"]
            boxes_coco = transformed["bboxes"]
            labels = transformed["labels"]
        else:
            image_tensor = transforms.ToTensor()(image_np)

        boxes_xyxy = [xywh_to_xyxy(box) for box in boxes_coco]

        target = {
            "image_id": torch.tensor(int(row["image_id"]), dtype=torch.int64),
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
