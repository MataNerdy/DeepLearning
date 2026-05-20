"""Data loading and corpus preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


QuestionRow = List[str]


def read_corpus(path: str | Path) -> list[QuestionRow]:
    """Read TSV rows: query, duplicate candidate, distractor candidates..."""
    path = Path(path)
    rows: list[QuestionRow] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = line.rstrip("\n").split("\t")
            if row and row[0]:
                rows.append(row)
    return rows


def build_training_corpus(rows: Sequence[QuestionRow], tokenizer, normalizer=None) -> list[list[str]]:
    """Build tokenized Word2Vec corpus from train TSV rows."""
    corpus: list[list[str]] = []
    for row in rows:
        if len(row) < 2:
            continue
        question, *candidates = row
        for candidate in candidates:
            tokens = tokenizer.tokenize(f"{question} {candidate}")
            if normalizer is not None:
                tokens = normalizer(tokens)
            if tokens:
                corpus.append(tokens)
    return corpus
