import argparse
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.load_data import DEFAULT_HF_DATASET, DEFAULT_SPLITS


def convert_bbox_xywh_to_yolo(box, img_w, img_h):
    x, y, w, h = [float(v) for v in box]
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def normalize_sequence(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return value


def prepare_split(df, split_name, output_dir):
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_names = set()
    bad_boxes = 0
    skipped_images = 0

    for _, row in df.iterrows():
        try:
            image_id = row["image_id"]
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            img_w, img_h = image.size
            image.save(image_dir / f"{image_id}.jpg")

            bboxes = normalize_sequence(row["objects"]["bbox"])
            labels = normalize_sequence(row["objects"]["category"])

            if isinstance(bboxes, list) and len(bboxes) > 0 and not isinstance(bboxes[0], (list, tuple, np.ndarray)):
                bboxes = [bboxes]
                labels = [labels]
            elif not isinstance(bboxes, list):
                bboxes, labels = [], []

            yolo_labels = []
            for box, cls in zip(bboxes, labels):
                try:
                    yolo_box = convert_bbox_xywh_to_yolo(box, img_w, img_h)
                    cls_id = int(cls) - 1
                    class_names.add(cls_id)
                    yolo_labels.append(f"{cls_id} {' '.join(f'{v:.6f}' for v in yolo_box)}")
                except Exception:
                    bad_boxes += 1

            with open(label_dir / f"{image_id}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_labels))
        except Exception:
            skipped_images += 1

    return class_names, bad_boxes, skipped_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/halo_yolo")
    parser.add_argument("--dataset-uri", type=str, default=DEFAULT_HF_DATASET)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.dataset_uri + DEFAULT_SPLITS["train"])
    df_val = pd.read_parquet(args.dataset_uri + DEFAULT_SPLITS["validation"])

    train_classes, train_bad, train_skip = prepare_split(df_train, "train", output_dir)
    val_classes, val_bad, val_skip = prepare_split(df_val, "val", output_dir)

    class_ids = sorted(train_classes | val_classes)
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_ids),
        "names": [f"class_{i}" for i in class_ids],
    }

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    print(f"Done: {output_dir}")
    print(f"Bad boxes: {train_bad + val_bad}")
    print(f"Skipped images: {train_skip + val_skip}")


if __name__ == "__main__":
    main()
