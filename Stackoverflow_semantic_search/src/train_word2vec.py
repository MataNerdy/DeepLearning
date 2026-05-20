"""Train Word2Vec embeddings on StackOverflow duplicate-question corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data import build_training_corpus, read_corpus
    from .embeddings import train_word2vec
    from .tokenization import TechnicalTokenizer, normalize_tokens
except ImportError:
    from data import build_training_corpus, read_corpus
    from embeddings import train_word2vec
    from tokenization import TechnicalTokenizer, normalize_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True, help="Path to train.tsv")
    parser.add_argument("--output", required=True, help="Where to save trained vectors")
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    tokenizer = TechnicalTokenizer()
    normalizer = normalize_tokens if args.normalize else None
    rows = read_corpus(args.train_path)
    corpus = build_training_corpus(rows, tokenizer, normalizer=normalizer)
    vectors = train_word2vec(
        corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    vectors.save_word2vec_format(args.output, binary=True)
    print(f"Saved vectors to {args.output}")


if __name__ == "__main__":
    main()
