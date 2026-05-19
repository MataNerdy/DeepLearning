from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    image_size: int = 256
    batch_size: int = 25
    epochs: int = 100
    lr: float = 1e-4
    seed: int = 42
    val_size: float = 0.20
    test_size: float = 0.20
    checkpoint_dir: Path = Path("checkpoints")
    report_dir: Path = Path("reports")
