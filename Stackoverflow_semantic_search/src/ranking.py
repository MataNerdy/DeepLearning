"""Candidate ranking for semantic duplicate-question retrieval."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .embeddings import question_to_vec

RankedCandidate = Tuple[int, str, float]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_candidates(question: str, candidates: list[str], embeddings, tokenizer, *, dim: int = 200) -> list[RankedCandidate]:
    """Return candidates sorted by cosine similarity to the query."""
    query_vector = question_to_vec(question, embeddings, tokenizer, dim=dim)
    ranked: list[RankedCandidate] = []

    for position, candidate in enumerate(candidates):
        candidate_vector = question_to_vec(candidate, embeddings, tokenizer, dim=dim)
        score = cosine_similarity(query_vector, candidate_vector)
        ranked.append((position, candidate, score))

    return sorted(ranked, key=lambda item: item[2], reverse=True)
