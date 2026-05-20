"""Tokenization utilities for StackOverflow semantic search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


class SimpleTokenizer:
    """Baseline tokenizer: keeps only \w+ tokens.

    This intentionally simple implementation is useful as a baseline, but it
    loses many programming-language tokens such as c++, c#, .net, node.js.
    """

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text)


@dataclass
class TechnicalTokenizer:
    """Tokenizer adapted for technical StackOverflow text.

    Keeps common programming tokens with symbols: c++, c#, .net, node.js,
    python-3.x, etc. This was the strongest engineering improvement in the
    original notebook experiments.
    """

    lower: bool = True
    pattern: str = r"[A-Za-z0-9_+#.-]+"

    def __post_init__(self) -> None:
        self._regex = re.compile(self.pattern)

    def tokenize(self, text: str) -> List[str]:
        tokens = self._regex.findall(text)
        return [token.lower() for token in tokens] if self.lower else tokens


def normalize_tokens(
    tokens: Iterable[str],
    *,
    stopwords: set[str] | None = None,
    lemmatizer=None,
    min_len: int = 2,
) -> List[str]:
    """Lowercase and optionally remove stopwords / lemmatize tokens."""
    stopwords = stopwords or set()
    token_re = re.compile(r"[A-Za-z0-9_+#.-]+")
    normalized: List[str] = []

    for token in tokens:
        token = token.lower()
        if not token_re.fullmatch(token):
            continue
        if token in stopwords:
            continue
        if lemmatizer is not None:
            token = lemmatizer.lemmatize(token)
        if len(token) < min_len:
            continue
        normalized.append(token)

    return normalized
