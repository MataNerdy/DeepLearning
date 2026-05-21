from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class MathProblemDataset(Dataset):
    """PyTorch dataset for mathematical problem topic classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        labels2ids: dict[str, int],
        tokenizer,
        text_column: str = "problem_text",
        label_column: str = "topic",
        max_length: int = 256,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.labels2ids = labels2ids
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[self.text_column])
        label = self.labels2ids[str(row[self.label_column])]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
