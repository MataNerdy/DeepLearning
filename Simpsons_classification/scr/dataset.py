from pathlib import Path
import pickle
from typing import Iterable, Optional

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_MODES, RESCALE_SIZE


def get_transforms(mode: str, image_size: int = RESCALE_SIZE):
    """Return torchvision transforms for train/val/test mode."""
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class SimpsonsDataset(Dataset):
    """Image dataset for Simpsons character classification."""

    def __init__(
        self,
        files: Iterable[Path],
        mode: str,
        label_encoder: Optional[LabelEncoder] = None,
        image_size: int = RESCALE_SIZE,
    ):
        super().__init__()
        self.files = sorted([Path(f) for f in files])
        self.mode = mode

        if self.mode not in DATA_MODES:
            raise ValueError(f"{self.mode} is incorrect; correct modes: {DATA_MODES}")

        self.transform = get_transforms(mode, image_size=image_size)
        self.label_encoder = label_encoder

        if self.mode != "test":
            self.labels = [path.parent.name for path in self.files]
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def load_sample(file: Path):
        return Image.open(file).convert("RGB")

    def __getitem__(self, index):
        image = self.load_sample(self.files[index])
        image = self.transform(image)

        if self.mode == "test":
            return image

        label = self.labels[index]
        target = self.label_encoder.transform([label])[0]
        return image, target


def save_label_encoder(label_encoder: LabelEncoder, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(label_encoder, file)


def load_label_encoder(path: str | Path) -> LabelEncoder:
    with open(path, "rb") as file:
        return pickle.load(file)
