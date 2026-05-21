from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
import requests
from sklearn.model_selection import train_test_split


DEFAULT_DATA_URL = "https://docs.google.com/spreadsheets/d/13YIbphbWc62sfa-bCh8MLQWKizaXbQK9/export?format=xlsx"


@dataclass
class DataConfig:
    source_url: str = DEFAULT_DATA_URL
    text_column: str = "problem_text"
    label_column: str = "topic"
    test_size: float = 0.2
    random_state: int = 42


def load_math_problem_data(config: DataConfig = DataConfig()) -> pd.DataFrame:
    """Load dataset from Google Sheets export URL and apply basic cleaning."""
    response = requests.get(config.source_url, timeout=60)
    response.raise_for_status()

    df = pd.read_excel(io.BytesIO(response.content), index_col=0)
    df = df.dropna(subset=[config.text_column, config.label_column])
    df = df.drop_duplicates().reset_index(drop=True)

    df[config.text_column] = df[config.text_column].astype(str)
    df[config.label_column] = df[config.label_column].astype(str)
    return df


def make_label_maps(df: pd.DataFrame, label_column: str = "topic") -> tuple[dict[str, int], dict[int, str]]:
    topics = sorted(df[label_column].unique().tolist())
    labels2ids = {label: idx for idx, label in enumerate(topics)}
    ids2labels = {idx: label for label, idx in labels2ids.items()}
    return labels2ids, ids2labels


def make_train_test_split(
    df: pd.DataFrame,
    config: DataConfig = DataConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df[config.label_column],
        random_state=config.random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
