"""RM3 pseudo-relevance feedback over BM25.

The first method in this project that is supposed to *beat* the baseline rather than
establish it. Everything measured so far — tokenisation, `k1`/`b` — was a property of
the baseline; this is a different retrieval strategy, and the benchmark exists to say
whether it earns its place.

**The idea.** A user's query is a short, impoverished description of an information
need. The top few documents BM25 returns are, if retrieval works at all, mostly about
that need — so the terms *they* use are evidence about which words the need is really
phrased in. Feed those terms back into the query and retrieve again. No relevance
judgements are involved, which is why it is *pseudo*-relevance feedback: the assumption
that the top documents are relevant is exactly the assumption that can go wrong, and
when it does the expansion drifts the query somewhere worse. That failure mode is the
reason this is an experiment rather than a default.

**The estimate.** Following Lavrenko and Croft's relevance models, and the RM3 variant
Anserini implements:

1. Retrieve, and keep the top ``fb_docs`` documents with their BM25 scores.
2. Weight those documents by normalised score, ``w_d = s_d / sum(s)``, so a document
   the first pass was confident about contributes more.
3. Estimate the relevance model over terms as
   ``P(t|R) = sum_d w_d * tf(t,d) / |d|``.
4. Keep the ``fb_terms`` highest-weighted terms and renormalise them to sum to 1.
5. Interpolate with the original query, which is the "3" in RM3 and the part that keeps
   the expansion anchored:
   ``P(t|Q') = (1 - alpha) * P(t|Q) + alpha * P(t|R)``
   with ``P(t|Q) = tf(t,Q) / |Q|``.
6. Score with the expanded, weighted query.

``alpha=0`` must therefore reproduce plain BM25 exactly, which is the strongest
available test of the whole construction and is asserted in ``tests/test_rm3.py``.

**Cost.** One extra retrieval pass per query, plus re-tokenising ``fb_docs`` documents.
Feedback documents are re-tokenised from the corpus rather than served from a stored
forward index: at ten documents per query that is free, and keeping a forward index over
171,332 TREC-COVID documents would not be.
"""

from __future__ import annotations

from collections.abc import Mapping

from tqdm import tqdm

from groundwork.retrieval.bm25 import BM25Retriever
from groundwork.retrieval.tokenize import Tokenizer


class RM3Retriever:
    """BM25 with RM3 pseudo-relevance feedback.

    Args:
        fb_docs: Feedback documents taken from the first pass.
        fb_terms: Expansion terms kept from the relevance model.
        alpha: Weight on the relevance model against the original query, in [0, 1].
            0 is plain BM25; 1 discards the original query entirely.
        k1: BM25 term frequency saturation.
        b: BM25 length normalisation.
        tokenizer: Tokeniser; defaults to Lucene stopwords with Porter stemming.
    """

    def __init__(
        self,
        fb_docs: int = 10,
        fb_terms: int = 10,
        alpha: float = 0.5,
        k1: float = 0.9,
        b: float = 0.4,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if fb_docs < 0:
            raise ValueError("fb_docs must be non-negative")
        if fb_terms < 0:
            raise ValueError("fb_terms must be non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.alpha = alpha
        self.bm25 = BM25Retriever(k1=k1, b=b, tokenizer=tokenizer)
        self._corpus: Mapping[str, Mapping[str, str]] | None = None

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokeniser in use, which is the BM25 index's."""
        return self.bm25.tokenizer

    def index(self, corpus: Mapping[str, Mapping[str, str]], show_progress: bool = True) -> None:
        """Build the BM25 index and keep a reference to the corpus for feedback.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``.
            show_progress: Display a progress bar.
        """
        self.bm25.index(corpus, show_progress=show_progress)
        self._corpus = corpus

    def _document_tokens(self, doc_id: str) -> list[str]:
        """Tokens of one document, re-tokenised on demand."""
        if self._corpus is None:
            raise RuntimeError("index() must be called before searching")
        fields = self._corpus[doc_id]
        title = fields.get("title", "") or ""
        text = fields.get("text", "") or ""
        return self.tokenizer(f"{title} {text}".strip())

    def relevance_model(self, feedback: list[tuple[str, float]]) -> dict[str, float]:
        """Estimate ``P(t|R)`` from scored feedback documents.

        Args:
            feedback: ``[(doc_id, score), ...]`` from the first retrieval pass.

        Returns:
            ``{term: weight}`` over the top ``fb_terms`` terms, summing to 1.0. Empty
            when there is no usable feedback.
        """
        if not feedback or self.fb_terms == 0:
            return {}

        total = sum(score for _, score in feedback)
        if total <= 0.0:
            return {}

        model: dict[str, float] = {}
        for doc_id, score in feedback:
            tokens = self._document_tokens(doc_id)
            if not tokens:
                continue
            doc_weight = score / total
            length = len(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for term, count in counts.items():
                model[term] = model.get(term, 0.0) + doc_weight * count / length

        if not model:
            return {}

        # Ties broken by term so the expansion is deterministic, matching the ranking
        # convention used everywhere else in the project.
        ranked = sorted(model.items(), key=lambda item: (-item[1], item[0]))[: self.fb_terms]
        mass = sum(weight for _, weight in ranked)
        if mass <= 0.0:
            return {}
        return {term: weight / mass for term, weight in ranked}

    def expanded_query(self, query: str) -> dict[str, float]:
        """Build the interpolated query that RM3 actually scores.

        Args:
            query: Original query text.

        Returns:
            ``{term: weight}``. With ``alpha=0`` this is the original query's term
            distribution and nothing else.
        """
        tokens = self.tokenizer(query)
        original: dict[str, float] = {}
        if tokens:
            for token in tokens:
                original[token] = original.get(token, 0.0) + 1.0 / len(tokens)

        if self.alpha == 0.0 or self.fb_docs == 0:
            return original

        first_pass = self.bm25.search(query, top_k=self.fb_docs)
        model = self.relevance_model(first_pass)
        if not model:
            return original

        expanded = {term: (1.0 - self.alpha) * weight for term, weight in original.items()}
        for term, weight in model.items():
            expanded[term] = expanded.get(term, 0.0) + self.alpha * weight
        return expanded

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Retrieve for ``query`` with pseudo-relevance feedback.

        Args:
            query: Query text.
            top_k: Maximum documents to return.

        Returns:
            ``[(doc_id, score), ...]``, highest first, ties broken by doc id.
        """
        weights = self.expanded_query(query)
        if not weights:
            return []
        scores = self.bm25.score_weighted_terms(weights)
        return self.bm25.rank_scores(scores, top_k)

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
        iterator = tqdm(items, desc="Retrieving (RM3)", unit="query") if show_progress else items
        return {qid: dict(self.search(text, top_k=top_k)) for qid, text in iterator}

    def with_parameters(
        self,
        fb_docs: int | None = None,
        fb_terms: int | None = None,
        alpha: float | None = None,
    ) -> RM3Retriever:
        """Return an RM3 retriever over this same index with different feedback settings.

        Shares the BM25 index and the corpus reference, so a sweep over feedback
        parameters costs one index build rather than one per cell.

        Args:
            fb_docs: Replacement feedback document count, or None to keep.
            fb_terms: Replacement expansion term count, or None to keep.
            alpha: Replacement interpolation weight, or None to keep.

        Returns:
            A retriever ready to search.

        Raises:
            RuntimeError: If this retriever has not been indexed.
        """
        if self._corpus is None:
            raise RuntimeError("index() must be called before with_parameters()")

        clone = RM3Retriever(
            fb_docs=self.fb_docs if fb_docs is None else fb_docs,
            fb_terms=self.fb_terms if fb_terms is None else fb_terms,
            alpha=self.alpha if alpha is None else alpha,
            k1=self.bm25.k1,
            b=self.bm25.b,
            tokenizer=self.bm25.tokenizer,
        )
        clone.bm25 = self.bm25
        clone._corpus = self._corpus
        return clone

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        base = self.bm25.describe()
        return {
            "method": "rm3",
            "base": base,
            "fb_docs": self.fb_docs,
            "fb_terms": self.fb_terms,
            "alpha": self.alpha,
            "k1": self.bm25.k1,
            "b": self.bm25.b,
            "tokenizer": base["tokenizer"],
        }
