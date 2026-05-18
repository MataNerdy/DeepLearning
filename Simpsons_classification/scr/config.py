from pathlib import Path
import torch

DATA_MODES = ["train", "val", "test"]
RESCALE_SIZE = 224
RANDOM_STATE = 42

# Kaggle default path. Can be overridden via CLI args.
DEFAULT_INPUT_DIR = Path("/kaggle/input/journey-springfield")
DEFAULT_TRAIN_DIR = DEFAULT_INPUT_DIR / "train" / "simpsons_dataset"
DEFAULT_TEST_DIR = DEFAULT_INPUT_DIR / "testset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
