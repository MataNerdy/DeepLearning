import argparse
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def load_ph2(dataset_dir: Path, image_size: int = 256):
    images, masks = [], []
    root = dataset_dir / "PH2 Dataset images"
    if not root.exists():
        raise FileNotFoundError(f"Expected folder not found: {root}. Extract PH2Dataset.rar first.")

    for current_root, _, files in os_walk_sorted(root):
        current_root = Path(current_root)
        if not files:
            continue
        if str(current_root).endswith("_Dermoscopic_Image"):
            images.append(imread(current_root / files[0]))
        elif str(current_root).endswith("_lesion"):
            masks.append(imread(current_root / files[0]))

    x = np.array([resize(img, (image_size, image_size), mode="constant", anti_aliasing=True) for img in images], dtype=np.float32)
    y = np.array([resize(mask, (image_size, image_size), mode="constant", anti_aliasing=False) > 0.5 for mask in masks], dtype=np.float32)
    return x, y


def os_walk_sorted(root: Path):
    import os
    for current_root, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        yield current_root, dirs, files


def make_splits(x, y, val_size=0.2, test_size=0.2, seed=42):
    idx = np.arange(len(x))
    train_idx, temp_idx = train_test_split(idx, test_size=val_size + test_size, random_state=seed, shuffle=True)
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=seed, shuffle=True)
    return {
        "x_train": x[train_idx], "y_train": y[train_idx],
        "x_val": x[val_idx], "y_val": y[val_idx],
        "x_test": x[test_idx], "y_test": y[test_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PH2 segmentation arrays.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/PH2Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = load_ph2(args.dataset_dir, args.image_size)
    splits = make_splits(x, y, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in splits.items():
        np.save(args.output_dir / f"{name}.npy", arr)
    print(f"Saved processed dataset to {args.output_dir}. Images: {len(x)}")


if __name__ == "__main__":
    main()
