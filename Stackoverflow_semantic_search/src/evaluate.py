"""Evaluate a Word2Vec semantic search model on StackOverflow duplicate ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

try:
    from .data import read_corpus
    from .embeddings import load_keyed_vectors
    from .metrics import metrics_table
    from .ranking import rank_candidates
    from .tokenization import SimpleTokenizer, TechnicalTokenizer
except ImportError:  # allows `python src/evaluate.py ...`
    from data import read_corpus
    from embeddings import load_keyed_vectors
    from metrics import metrics_table
    from ranking import rank_candidates
    from tokenization import SimpleTokenizer, TechnicalTokenizer


def get_tokenizer(name: str):
    if name == "simple":
        return SimpleTokenizer()
    if name == "technical":
        return TechnicalTokenizer()
    raise ValueError(f"Unknown tokenizer: {name}")


def evaluate(validation_path: str | Path, embeddings_path: str | Path, *, tokenizer_name: str = "technical", max_examples: int = 1000, dim: int = 200):
    data = read_corpus(validation_path)
    embeddings = load_keyed_vectors(embeddings_path, binary=True)
    tokenizer = get_tokenizer(tokenizer_name)

    duplicate_ranks: list[int] = []
    for i, row in enumerate(tqdm(data[:max_examples], desc="Ranking validation queries")):
        question, *candidates = row
        ranked = rank_candidates(question, candidates, embeddings, tokenizer, dim=dim)
        duplicate_rank = [item[0] for item in ranked].index(0) + 1
        duplicate_ranks.append(duplicate_rank)

    return metrics_table(duplicate_ranks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-path", required=True, help="Path to validation.tsv")
    parser.add_argument("--embeddings-path", required=True, help="Path to SO_vectors_200.bin")
    parser.add_argument("--tokenizer", choices=["simple", "technical"], default="technical")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = evaluate(
        args.validation_path,
        args.embeddings_path,
        tokenizer_name=args.tokenizer,
        max_examples=args.max_examples,
        dim=args.dim,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
