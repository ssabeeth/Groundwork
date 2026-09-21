"""Reciprocal rank fusion.

Combining two rankings is mostly a problem of units. BM25 scores are unbounded sums of
IDF terms; cosine similarities live in [-1, 1]. Adding them means inventing a conversion,
and every conversion — min-max, z-score, dividing by the top score — is a decision that
moves the result and has to be tuned and recorded.

Reciprocal rank fusion sidesteps this by discarding scores and keeping only positions:

    RRF(d) = sum over systems of 1 / (k + rank(d))

with ranks starting at 1. A document ranked first contributes ``1/(k+1)`` no matter what
score produced it, so no normalisation is needed and no system can dominate by having a
larger scale. The cost is real: a document a system is wildly confident about is treated
identically to one it barely preferred, and genuine confidence information is thrown
away. Which trade wins is empirical, and this repo has the harness to answer it.

``k`` controls how quickly the contribution decays with rank. Cormack, Clarke and
Buettcher (2009) proposed 60 and it is widely used as a default, but it is a parameter
like any other: it moves results, so it is swept on a training split and recorded, not
assumed.
"""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_K = 60.0

Run = Mapping[str, Mapping[str, float]]


def _ranked_ids(scores: Mapping[str, float]) -> list[str]:
    """Document ids best first, ties broken by id, matching the project convention."""
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def reciprocal_rank_fusion(
    runs: list[Run],
    k: float = DEFAULT_K,
    top_k: int = 100,
    weights: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Fuse several runs by reciprocal rank.

    Only the union of each query's retrieved documents is considered. A document one
    system never returned contributes nothing from that system rather than being scored
    as if ranked last — treating an unseen document as ranked at infinity is the same
    thing, and doing it explicitly keeps the arithmetic honest.

    Args:
        runs: Runs to fuse, each ``{query_id: {doc_id: score}}``. They need not cover
            the same queries; the union is used.
        k: Rank decay constant. Larger flattens the contribution of top ranks.
        top_k: Documents to keep per query.
        weights: Per-run multipliers, defaulting to 1.0 each. A weight is another
            parameter that must be tuned on a training split if it is used at all.

    Returns:
        ``{query_id: {doc_id: fused_score}}``.

    Raises:
        ValueError: If no runs are given, ``k`` is negative, ``top_k`` is not positive,
            or ``weights`` does not match ``runs`` in length.
    """
    if not runs:
        raise ValueError("at least one run is required")
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError(f"{len(weights)} weights for {len(runs)} runs")

    query_ids = sorted({qid for run in runs for qid in run})

    fused: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        totals: dict[str, float] = {}
        for run, weight in zip(runs, weights, strict=True):
            if weight == 0.0:
                # Otherwise the run's documents enter the ranking at score 0.0, which
                # contradicts both the intent of a zero weight and the project's rule
                # that a document contributing nothing is not ranked.
                continue
            scores = run.get(query_id)
            if not scores:
                continue
            for rank, doc_id in enumerate(_ranked_ids(scores), start=1):
                totals[doc_id] = totals.get(doc_id, 0.0) + weight / (k + rank)
        ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        fused[query_id] = dict(ranked)
    return fused
