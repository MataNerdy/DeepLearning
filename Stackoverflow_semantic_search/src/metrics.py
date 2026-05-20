"""Ranking metrics used in the StackOverflow retrieval experiment."""

from __future__ import annotations

import numpy as np


def hits_at_k(duplicate_ranks: list[int], k: int) -> float:
    """Share of queries where the duplicate was ranked in top-k."""
    ranks = np.asarray(duplicate_ranks)
    return float(np.mean((ranks <= k).astype(float)))


def dcg_at_k(duplicate_ranks: list[int], k: int) -> float:
    """Discounted cumulative gain for one relevant duplicate per query."""
    ranks = np.asarray(duplicate_ranks)
    gains = (1 / np.log2(1 + ranks)) * (ranks <= k)
    return float(np.mean(gains))


def metrics_table(duplicate_ranks: list[int], ks=(1, 5, 10, 100, 500, 1000)) -> list[dict[str, float | int]]:
    return [
        {"k": k, "hits": hits_at_k(duplicate_ranks, k), "dcg": dcg_at_k(duplicate_ranks, k)}
        for k in ks
    ]
