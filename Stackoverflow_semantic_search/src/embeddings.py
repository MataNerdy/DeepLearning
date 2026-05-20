"""Embedding loading, training and sentence vectorization."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def load_keyed_vectors(path: str | Path, *, binary: bool = True) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(str(path), binary=binary)


def train_word2vec(
    corpus: Sequence[Sequence[str]],
    *,
    vector_size: int = 200,
    window: int = 12,
    min_count: int = 5,
    workers: int = 4,
    seed: int = 42,
) -> KeyedVectors:
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )
    return model.wv


def question_to_vec(question: str, embeddings, tokenizer, *, dim: int = 200) -> np.ndarray:
    """Represent a question as the mean of known token vectors."""
    vectors = []
    for token in tokenizer.tokenize(question):
        if token in embeddings:
            vectors.append(embeddings[token])
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0)
