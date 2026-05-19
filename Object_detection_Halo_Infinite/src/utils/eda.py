from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_bbox_size_histogram(widths, heights, bins=30):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=bins, edgecolor="black")
    plt.title("BBox Widths Distribution")
    plt.xlabel("Width (px)")
    plt.ylabel("Count")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=bins, edgecolor="black")
    plt.title("BBox Heights Distribution")
    plt.xlabel("Height (px)")
    plt.ylabel("Count")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def dataset_statistics(dataset):
    num_images = len(dataset)
    all_labels, all_widths, all_heights, bboxes_per_image = [], [], [], []

    for idx in tqdm(range(num_images), desc="Collecting stats"):
        _, target = dataset[idx]
        boxes = target["boxes"]
        labels = target["labels"]

        if boxes.numel() > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            all_widths.extend(widths.tolist())
            all_heights.extend(heights.tolist())
            all_labels.extend(labels.tolist())

        bboxes_per_image.append(len(labels))

    print(f"Images: {num_images}")
    print(f"Total objects: {len(all_labels)}")
    print(
        f"Objects per image: min={min(bboxes_per_image)}, "
        f"max={max(bboxes_per_image)}, mean={np.mean(bboxes_per_image):.2f}"
    )

    label_counts = Counter(all_labels)
    print("Object counts by class:")
    for label, count in label_counts.items():
        print(f"  Class {label}: {count}")

    if all_widths:
        print("\nBBox sizes (width x height):")
        print(f"  Mean width: {np.mean(all_widths):.2f} | Mean height: {np.mean(all_heights):.2f}")
        print(f"  Min width: {np.min(all_widths):.2f} | Max width: {np.max(all_widths):.2f}")
        print(f"  Min height: {np.min(all_heights):.2f} | Max height: {np.max(all_heights):.2f}")
        plot_bbox_size_histogram(all_widths, all_heights)
