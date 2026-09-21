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

Documents are indexed as ``title + " " + text``: one field, one bag of words.

This does **not** match how BEIR produced its published baselines, and the docstring
here previously claimed it did. BEIR used Anserini and, in its own words, "index the
title (if available) and passage as separate fields for documents" — so a query is
scored against each field with its own length normalisation and its own IDF, and the
scores are summed. :class:`MultiFieldBM25Retriever` implements that arrangement, and
experiment 10 measures what the difference is worth. Concatenation is kept as the
default because every result in this repository so far was produced with it.
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

    def with_parameters(self, k1: float | None = None, b: float | None = None) -> BM25Retriever:
        """Return a retriever over this same index, scoring with different parameters.

        The inverted index, document lengths and IDF values are functions of the corpus
        and the tokeniser alone — ``k1`` and ``b`` appear only in scoring. A parameter
        sweep can therefore reuse one index across every cell, which on SciFact is the
        difference between fifteen seconds and half an hour.

        The index is shared, not copied. Neither retriever mutates it during scoring, so
        this is safe, but re-indexing one does not affect the other.

        Args:
            k1: Replacement term-frequency saturation, or None to keep this one's.
            b: Replacement length normalisation, or None to keep this one's.

        Returns:
            A retriever ready to search, with no indexing required.

        Raises:
            RuntimeError: If this retriever has not been indexed yet.
            ValueError: If the replacement parameters are out of range.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before with_parameters()")

        clone = BM25Retriever(
            k1=self.k1 if k1 is None else k1,
            b=self.b if b is None else b,
            tokenizer=self.tokenizer,
        )
        clone.doc_ids = self.doc_ids
        clone.doc_lengths = self.doc_lengths
        clone.avg_doc_length = self.avg_doc_length
        clone._postings = self._postings
        clone._idf = self._idf
        return clone

    def _length_factor(self) -> np.ndarray:
        """Per-document denominator term: ``k1 * (1 - b + b * |D| / avgdl)``."""
        return self.k1 * (1.0 - self.b + self.b * (self.doc_lengths / self.avg_doc_length))

    def score_query(self, query: str) -> np.ndarray:
        """Score every document against ``query``, in :attr:`doc_ids` order.

        Exposed so a multi-field retriever can hold one index per field and add their
        scores, which is how Lucene scores a query against several fields.

        Args:
            query: Query text.

        Returns:
            One score per document.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before score_query()")
        return self._score_all(query)

    def _score_all(self, query: str) -> np.ndarray:
        """Score every document against ``query``."""
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        if self.avg_doc_length == 0.0:
            return scores

        length_factor = self._length_factor()

        for term in self.tokenizer(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            indices, freqs = posting
            numerator = freqs * (self.k1 + 1.0)
            denominator = freqs + length_factor[indices]
            scores[indices] += self._idf[term] * (numerator / denominator)

        return scores

    def score_weighted_terms(self, term_weights: Mapping[str, float]) -> np.ndarray:
        """Score every document against already-tokenised terms carrying weights.

        Plain :meth:`search` treats a query as a bag of equally weighted terms, repeated
        terms counting twice. Query expansion needs the general case: terms with
        arbitrary non-negative weights, already tokenised, because an expansion term is
        chosen from the index and must not be re-tokenised on the way back in.

        Args:
            term_weights: ``{term: weight}``, terms already tokenised. Terms absent
                from the index are ignored, as in ordinary scoring.

        Returns:
            A score per document, in :attr:`doc_ids` order.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before score_weighted_terms()")

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        if self.avg_doc_length == 0.0:
            return scores

        length_factor = self._length_factor()

        for term, weight in term_weights.items():
            posting = self._postings.get(term)
            if posting is None or weight == 0.0:
                continue
            indices, freqs = posting
            numerator = freqs * (self.k1 + 1.0)
            denominator = freqs + length_factor[indices]
            scores[indices] += weight * self._idf[term] * (numerator / denominator)

        return scores

    def rank_scores(self, scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Turn a score vector into a ranking, dropping zeros and breaking ties by id.

        Shared by :meth:`search` and by any retriever that scores through
        :meth:`score_weighted_terms`, so every ranking in the project is built the same
        way and the tie-breaking rule lives in one place.

        Args:
            scores: One score per document, in :attr:`doc_ids` order.
            top_k: Maximum documents to return.

        Returns:
            ``[(doc_id, score), ...]``, highest first, ties broken by doc id ascending.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        nonzero = np.flatnonzero(scores > 0.0)
        if nonzero.size == 0:
            return []

        if nonzero.size > top_k:
            partition = np.argpartition(-scores[nonzero], top_k)[:top_k]
            nonzero = nonzero[partition]

        results = [(self.doc_ids[i], float(scores[i])) for i in nonzero]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]

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

        return self.rank_scores(self._score_all(query), top_k)

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

    def query_statistics(self, query: str) -> dict[str, float]:
        """Lexical properties of a query, computed from the index alone.

        These are the predictors the query-type analysis uses, and the reason they are
        computed here is that they must be knowable *before* any retrieval is run. A
        property derived from how well a system did on a query would make any
        correlation with that system's performance circular.

        IDF is the natural measure of how much signal an exact match carries: a term
        appearing in almost every document distinguishes nothing, while a rare one —
        a gene symbol, an accession number — is close to a unique key. Terms absent
        from the index are counted separately rather than scored as maximally rare,
        because a term no document contains gives BM25 nothing to match on.

        Args:
            query: Query text, tokenised with this index's tokeniser.

        Returns:
            ``max_idf``, ``mean_idf``, ``num_terms``, ``num_in_vocabulary`` and
            ``out_of_vocabulary_rate``. IDF statistics are 0.0 when no query term
            appears in the index.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before query_statistics()")

        tokens = self.tokenizer(query)
        known = [self._idf[token] for token in tokens if token in self._idf]
        return {
            "max_idf": max(known) if known else 0.0,
            "mean_idf": sum(known) / len(known) if known else 0.0,
            "num_terms": float(len(tokens)),
            "num_in_vocabulary": float(len(known)),
            "out_of_vocabulary_rate": ((len(tokens) - len(known)) / len(tokens) if tokens else 0.0),
        }

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "bm25",
            "variant": "lucene",
            "k1": self.k1,
            "b": self.b,
            "tokenizer": self.tokenizer.describe(),
        }


class MultiFieldBM25Retriever:
    """BM25 over several document fields scored separately, as Anserini indexes them.

    The single-field :class:`BM25Retriever` glues title and body into one bag of words.
    Lucene, and therefore Anserini and therefore BEIR's published baselines, does
    something different: each field is its own index with its own document lengths, its
    own average length and its own document frequencies, and a query scores against each
    and the results are added.

    The difference is not cosmetic, and it is largest exactly where documents are
    lopsided. A title-only document — 24.6% of TREC-COVID — is a very short document
    under concatenation, so length normalisation inflates whatever it does match. Split
    into fields, its title is an ordinary-length title and its body is empty and
    contributes nothing.

    Args:
        fields: Document fields to index, in order.
        weights: Per-field multipliers, defaulting to 1.0 each. Anserini's BEIR setup
            weights fields equally; anything else is a parameter that must be tuned on
            a training split and recorded.
        k1: Term frequency saturation, shared across fields.
        b: Length normalisation, shared across fields.
        tokenizer: Tokeniser; defaults to Lucene stopwords with Porter stemming.
    """

    def __init__(
        self,
        fields: tuple[str, ...] = ("title", "text"),
        weights: tuple[float, ...] | None = None,
        k1: float = 0.9,
        b: float = 0.4,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if not fields:
            raise ValueError("at least one field is required")
        if weights is not None and len(weights) != len(fields):
            raise ValueError(f"{len(weights)} weights for {len(fields)} fields")

        self.fields = fields
        self.weights = weights if weights is not None else tuple(1.0 for _ in fields)
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self._indexes: dict[str, BM25Retriever] = {}
        self.doc_ids: list[str] = []

    def index(self, corpus: Mapping[str, Mapping[str, str]], show_progress: bool = True) -> None:
        """Build one index per field.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``.
            show_progress: Display a progress bar per field.
        """
        if not corpus:
            raise ValueError("corpus is empty")

        self.doc_ids = list(corpus)
        self._indexes = {}
        for field in self.fields:
            # A view exposing only this field, so the single-field indexer sees it alone.
            view = {
                doc_id: {"title": "", "text": fields.get(field, "") or ""}
                for doc_id, fields in corpus.items()
            }
            retriever = BM25Retriever(k1=self.k1, b=self.b, tokenizer=self.tokenizer)
            retriever.index(view, show_progress=show_progress)
            self._indexes[field] = retriever

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return the ``top_k`` highest scoring documents, summing over fields.

        Args:
            query: Query text.
            top_k: Maximum documents to return.

        Returns:
            ``[(doc_id, score), ...]``, highest first, ties broken by doc id.
        """
        if not self._indexes:
            raise RuntimeError("index() must be called before search()")

        total = None
        for field, weight in zip(self.fields, self.weights, strict=True):
            scores = self._indexes[field].score_query(query) * weight
            total = scores if total is None else total + scores
        return self._indexes[self.fields[0]].rank_scores(total, top_k)

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
            "variant": "lucene-multifield",
            "fields": list(self.fields),
            "field_weights": list(self.weights),
            "k1": self.k1,
            "b": self.b,
            "tokenizer": self.tokenizer.describe(),
        }
