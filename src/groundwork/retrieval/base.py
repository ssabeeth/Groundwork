"""The interface every retriever implements.

Keeping this narrow is what makes the ablation possible: dense, hybrid and reranked
retrievers all produce a run in the same shape, so the evaluation harness never needs
to know which one it is scoring.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Anything that can turn queries into a scored ranking of document ids."""

    def index(self, corpus: Mapping[str, Mapping[str, str]]) -> None:
        """Build whatever structures are needed to search ``corpus``.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``.
        """
        ...

    def retrieve(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> dict[str, dict[str, float]]:
        """Score documents for each query.

        Args:
            queries: ``{query_id: query_text}``.
            top_k: Number of documents to return per query.

        Returns:
            ``{query_id: {doc_id: score}}``, at most ``top_k`` documents per query.
        """
        ...
