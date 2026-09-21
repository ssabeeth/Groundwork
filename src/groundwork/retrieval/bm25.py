"""BM25, in the form Lucene implements it.

Written out rather than pulled from a library because the point of the project is that
the baseline is understood and testable, and because the common pip BM25 packages use
the Robertson IDF, which can go negative for terms appearing in more than half the
corpus. Lucene's variant cannot, and the published BEIR baselines are Lucene's.

Scoring, for query term ``q`` and document ``D``::

    idf(q)   = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
    score    = sum_q idf(q) * (tf(q,D) * (k1 + 1))
                             / (tf(q,D) + k1 * (1 - b + b * |D| / avgdl))

Defaults are ``k1=0.9, b=0.4``, which is what the BEIR paper used for its BM25
baselines. Anserini's defaults are ``k1=0.9, b=0.4`` for BEIR too, but the Lucene
out-of-the-box defaults are ``k1=1.2, b=0.75`` — results are not comparable across
those settings, so the values used are always recorded with the run.

Documents are indexed as ``title + " " + text``, matching BEIR's convention. Doing
anything else, such as dropping the title, materially changes the numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from tqdm import tqdm

from groundwork.retrieval.tokenize import Tokenizer


class BM25Retriever:
    """Sparse lexical retrieval with Lucene-style BM25.

    Args:
        k1: Term frequency saturation. Higher means repeated terms keep adding score.
        b: Length normalisation, in [0, 1]. 0 disables it entirely.
        tokenizer: Tokeniser to use. Defaults to Lucene stopwords with Porter stemming.
    """

    def __init__(
        self,
        k1: float = 0.9,
        b: float = 0.4,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")

        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()

        self.doc_ids: list[str] = []
        self.doc_lengths: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avg_doc_length: float = 0.0
        # term -> (doc index array, term frequency array)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}

    def index(self, corpus: Mapping[str, Mapping[str, str]], show_progress: bool = True) -> None:
        """Build the inverted index over ``corpus``.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``.
            show_progress: Display a progress bar.
        """
        if not corpus:
            raise ValueError("corpus is empty")

        self.doc_ids = list(corpus)
        num_docs = len(self.doc_ids)
        lengths = np.zeros(num_docs, dtype=np.float32)

        # term -> {doc index: term frequency}
        staging: dict[str, dict[int, int]] = {}

        iterator = enumerate(self.doc_ids)
        if show_progress:
            iterator = tqdm(iterator, total=num_docs, desc="Indexing", unit="doc")

        for doc_index, doc_id in iterator:
            fields = corpus[doc_id]
            title = fields.get("title", "") or ""
            text = fields.get("text", "") or ""
            tokens = self.tokenizer(f"{title} {text}".strip())
            lengths[doc_index] = len(tokens)

            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                staging.setdefault(token, {})[doc_index] = count

        self.doc_lengths = lengths
        self.avg_doc_length = float(lengths.mean()) if num_docs else 0.0

        self._postings = {}
        self._idf = {}
        for term, doc_counts in staging.items():
            indices = np.fromiter(doc_counts.keys(), dtype=np.int32, count=len(doc_counts))
            freqs = np.fromiter(doc_counts.values(), dtype=np.float32, count=len(doc_counts))
            order = np.argsort(indices)
            self._postings[term] = (indices[order], freqs[order])
            df = len(doc_counts)
            self._idf[term] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def _score_all(self, query: str) -> np.ndarray:
        """Score every document against ``query``."""
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        if self.avg_doc_length == 0.0:
            return scores

        # Denominator length factor, per document: k1 * (1 - b + b * |D| / avgdl)
        length_factor = self.k1 * (1.0 - self.b + self.b * (self.doc_lengths / self.avg_doc_length))

        for term in self.tokenizer(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            indices, freqs = posting
            numerator = freqs * (self.k1 + 1.0)
            denominator = freqs + length_factor[indices]
            scores[indices] += self._idf[term] * (numerator / denominator)

        return scores

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return the ``top_k`` highest scoring documents for ``query``.

        Documents scoring exactly zero are omitted: they share no query term, and
        padding the ranking with them would only change tie-breaking noise.

        Args:
            query: Query text.
            top_k: Maximum documents to return.

        Returns:
            ``[(doc_id, score), ...]``, highest first, ties broken by doc id ascending.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before search()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = self._score_all(query)
        nonzero = np.flatnonzero(scores > 0.0)
        if nonzero.size == 0:
            return []

        if nonzero.size > top_k:
            partition = np.argpartition(-scores[nonzero], top_k)[:top_k]
            nonzero = nonzero[partition]

        results = [(self.doc_ids[i], float(scores[i])) for i in nonzero]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]

    def retrieve(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Score documents for every query.

        Args:
            queries: ``{query_id: query_text}``.
            top_k: Documents per query.
            show_progress: Display a progress bar.

        Returns:
            ``{query_id: {doc_id: score}}``.
        """
        items = list(queries.items())
        iterator = tqdm(items, desc="Retrieving", unit="query") if show_progress else items
        return {qid: dict(self.search(text, top_k=top_k)) for qid, text in iterator}

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "bm25",
            "variant": "lucene",
            "k1": self.k1,
            "b": self.b,
            "tokenizer": self.tokenizer.describe(),
        }
