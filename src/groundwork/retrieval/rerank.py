"""Cross-encoder reranking over a candidate set.

A bi-encoder embeds query and document separately, so it can precompute the whole
corpus and compare by dot product — cheap, and unable to let a query term influence how
a document is read. A cross-encoder concatenates the pair and runs the model over both
together, so every query word can attend to every document word. That is far more
accurate and hopelessly more expensive: the model runs once per *pair*, so nothing can
be precomputed and the cost scales with candidates times queries. It is therefore only
ever used to reorder a shortlist that something cheaper produced.

**The ceiling is the part to keep in view.** Reranking reorders; it cannot introduce a
document the first stage never retrieved. So recall@depth of the candidate run is a hard
upper bound on the reranker's recall, and a bound on how much nDCG it can recover. This
project has measured those ceilings, which is why reranking is run against the fused
candidates rather than BM25's: on NFCorpus, fusion's recall@100 is 0.3217 against BM25's
0.2461, and no reranker closes that gap.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

Run = Mapping[str, Mapping[str, float]]


def _require_cross_encoder():  # noqa: ANN202 - third-party class, not ours to name
    """Import the cross-encoder, or explain how to install it."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reranking requires sentence-transformers. Install with: pip install -e '.[dense]'"
        ) from exc
    return CrossEncoder


class CrossEncoderReranker:
    """Rescore the top candidates of an existing run with a cross-encoder.

    Args:
        model_name: Cross-encoder model id, recorded with every result.
        depth: Candidates per query to rescore. Documents below this keep their original
            order beneath the reranked block, so the run stays the same length.
        batch_size: Pair batch size. Affects speed only.
        max_length: Token cap for the concatenated pair, or None for the model default.
        device: Torch device string, or None to let the library choose.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        depth: int = 100,
        batch_size: int = 64,
        max_length: int | None = None,
        device: str | None = None,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.model_name = model_name
        self.depth = depth
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None

    @property
    def model(self):  # noqa: ANN201 - third-party class, not ours to name
        """The cross-encoder, loaded on first use."""
        if self._model is None:
            CrossEncoder = _require_cross_encoder()
            kwargs = {"device": self.device}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def _document_text(self, corpus: Mapping[str, Mapping[str, str]], doc_id: str) -> str:
        fields = corpus[doc_id]
        title = fields.get("title", "") or ""
        text = fields.get("text", "") or ""
        return f"{title} {text}".strip()

    def rerank(
        self,
        run: Run,
        queries: Mapping[str, str],
        corpus: Mapping[str, Mapping[str, str]],
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Reorder each query's top ``depth`` candidates.

        Scores below the reranked block are rewritten so the untouched tail sorts
        beneath every reranked document. Without that, a raw cross-encoder score — which
        can be negative — could place an unreranked document above a reranked one, and
        the ranking would no longer mean what it says.

        Args:
            run: ``{query_id: {doc_id: score}}`` from a first-stage retriever.
            queries: ``{query_id: query_text}``.
            corpus: ``{doc_id: {"title": ..., "text": ...}}``.
            show_progress: Display a progress bar over pairs.

        Returns:
            ``{query_id: {doc_id: score}}``, same queries and documents, reordered.
        """
        pairs: list[tuple[str, str]] = []
        layout: list[tuple[str, list[str], list[str]]] = []

        for query_id, scores in run.items():
            ranked = [
                doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
            head, tail = ranked[: self.depth], ranked[self.depth :]
            layout.append((query_id, head, tail))
            pairs.extend(
                (queries[query_id], self._document_text(corpus, doc_id)) for doc_id in head
            )

        if not pairs:
            return {query_id: {} for query_id in run}

        logger.info("Reranking %d pairs with %s", len(pairs), self.model_name)
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=show_progress
        )

        reranked: dict[str, dict[str, float]] = {}
        cursor = 0
        for query_id, head, tail in layout:
            head_scores = [float(scores[cursor + i]) for i in range(len(head))]
            cursor += len(head)
            ordered = sorted(zip(head, head_scores, strict=True), key=lambda kv: (-kv[1], kv[0]))
            result = dict(ordered)
            if tail:
                floor = min(head_scores) if head_scores else 0.0
                # Strictly below the reranked block, preserving the tail's own order.
                for offset, doc_id in enumerate(tail, start=1):
                    result[doc_id] = floor - offset
            reranked[query_id] = result
        return reranked

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "cross-encoder-rerank",
            "model": self.model_name,
            "depth": self.depth,
            "max_length": self.max_length,
        }
