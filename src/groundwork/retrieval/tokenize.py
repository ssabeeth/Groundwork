"""Tokenisation.

Tokenisation is the single biggest source of BM25 numbers that do not reproduce.
Stemming alone moves nDCG@10 on BEIR datasets by a point or two, and stopword handling
changes which documents are reachable at all. So the choices are explicit and
configurable rather than buried, and every result records which settings produced it.

The default stopword list is Lucene's 33-word English set, because the published BEIR
BM25 baselines were produced with Elasticsearch.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Lucene's ENGLISH_STOP_WORDS_SET.
LUCENE_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
        "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
        "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
    }
)  # fmt: skip


def porter_stemmer() -> Callable[[str], str]:
    """Return a Porter stemmer, or raise if ``snowballstemmer`` is not installed.

    Stemming is optional so the core package has no hard dependency beyond numpy.
    Install it with ``pip install groundwork[stem]``.
    """
    try:
        import snowballstemmer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Stemming requires snowballstemmer. Install with: pip install -e '.[stem]'"
        ) from exc

    stemmer = snowballstemmer.stemmer("english")
    return stemmer.stemWord


class Tokenizer:
    """Lowercase, split on alphanumeric runs, optionally drop stopwords and stem.

    Args:
        stopwords: Words to drop. Pass an empty set to keep everything.
        stem: Apply Porter stemming. Requires the ``stem`` extra.
        min_length: Drop tokens shorter than this, after stemming.
    """

    def __init__(
        self,
        stopwords: Iterable[str] | None = LUCENE_ENGLISH_STOPWORDS,
        stem: bool = True,
        min_length: int = 1,
    ) -> None:
        self.stopwords = frozenset(stopwords) if stopwords is not None else frozenset()
        self.stem = stem
        self.min_length = min_length
        self._stemmer = porter_stemmer() if stem else None

    def __call__(self, text: str) -> list[str]:
        """Tokenise ``text``."""
        tokens = TOKEN_PATTERN.findall(text.lower())
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        if self._stemmer is not None:
            tokens = [self._stemmer(token) for token in tokens]
        if self.min_length > 1:
            tokens = [token for token in tokens if len(token) >= self.min_length]
        return tokens

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "stopwords": "lucene_english" if self.stopwords else "none",
            "stem": "porter" if self.stem else "none",
            "min_length": self.min_length,
        }
