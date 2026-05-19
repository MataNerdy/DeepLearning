import pandas as pd

DEFAULT_SPLITS = {
    "train": "data/train-00000-of-00001-0d6632d599c29801.parquet",
    "validation": "data/validation-00000-of-00001-c6b77a557eeedd52.parquet",
}

DEFAULT_HF_DATASET = "hf://datasets/Francesco/halo-infinite-angel-videogame/"


def load_halo_split(split: str, base_uri: str = DEFAULT_HF_DATASET) -> pd.DataFrame:
    if split not in DEFAULT_SPLITS:
        raise ValueError(f"Unknown split: {split}. Available: {list(DEFAULT_SPLITS)}")
    return pd.read_parquet(base_uri + DEFAULT_SPLITS[split])
