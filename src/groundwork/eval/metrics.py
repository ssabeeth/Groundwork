"""Retrieval metrics, matching ``trec_eval`` conventions.

Two conventions are worth stating explicitly, because they are the usual source of
numbers that do not line up with published results:

1. **Gain function.** ``trec_eval``'s ``ndcg_cut`` uses an exponential gain,
   ``2**rel - 1``. For binary qrels (SciFact) this is identical to using ``rel``
   directly, since ``2**1 - 1 == 1``. For graded qrels (TREC-COVID and NFCorpus use
   levels 0/1/2) the two differ, and the exponential form weights a level-2 document
   three times a level-1 one rather than twice. ``gain`` selects between them.

2. **Ideal ordering.** The IDCG denominator is built from *all* judged-relevant
   documents for the query, not only those the system retrieved. A system that misses
   relevant documents entirely is penalised, which is the point.

Unjudged documents are treated as non-relevant (``rel = 0``), which is what
``trec_eval`` does and what the BEIR leaderboard assumes.

Ties in retrieval score are broken by document id (ascending) so that a run is scored
deterministically regardless of dict ordering upstream.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]

GainFn = Literal["exponential", "linear"]


def _gain(rel: float, gain: GainFn) -> float:
    """Gain contributed by a document at relevance level ``rel``."""
    if rel <= 0:
        return 0.0
    if gain == "exponential":
        return (2.0**rel) - 1.0
    return float(rel)


def _ranked_doc_ids(scores: Mapping[str, float], k: int) -> list[str]:
    """Top ``k`` document ids, highest score first, ties broken by id ascending."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in ordered[:k]]


def ndcg_at_k(
    ranked: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
    gain: GainFn = "exponential",
) -> float:
    """nDCG@k for a single query.

    Args:
        ranked: Document ids in rank order, best first. May be longer than ``k``.
        relevance: Relevance level per document id. Missing ids count as 0.
        k: Rank cutoff.
        gain: Gain function; see module docstring.

    Returns:
        nDCG@k in [0, 1]. Returns 0.0 when the query has no relevant documents,
        which matches ``trec_eval``'s behaviour of scoring such queries as zero
        rather than skipping them.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    dcg = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        rel = relevance.get(doc_id, 0)
        if rel > 0:
            dcg += _gain(rel, gain) / math.log2(rank + 1)

    ideal_levels = sorted((rel for rel in relevance.values() if rel > 0), reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal_levels[:k], start=1):
        idcg += _gain(rel, gain) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    ranked: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    """Recall@k for a single query: fraction of relevant documents found by rank ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = {doc_id for doc_id, rel in relevance.items() if rel > 0}
    if not relevant:
        return 0.0

    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / len(relevant)


def evaluate_run_per_query(
    run: Run,
    qrels: Qrels,
    k_values: Iterable[int] = (10, 100),
    gain: GainFn = "exponential",
) -> dict[str, dict[str, float]]:
    """Score a run without averaging: one metric dict per query.

    This is what a paired significance test needs. Averages throw away the pairing
    between two systems on the same query, and the pairing is where most of the
    statistical power lives — two systems can differ by a tenth of a point on the mean
    while disagreeing substantially on individual queries, or vice versa.

    Queries present in ``qrels`` but absent from ``run`` are scored as 0 rather than
    dropped, exactly as in :func:`evaluate_run`.

    Args:
        run: ``{query_id: {doc_id: score}}``. Higher score means higher rank.
        qrels: ``{query_id: {doc_id: relevance_level}}``.
        k_values: Rank cutoffs to report.
        gain: Gain function for nDCG.

    Returns:
        ``{query_id: {"ndcg@10": ..., "recall@10": ...}}``, one entry per query in
        ``qrels``, ordered by query id.
    """
    k_list = sorted(set(k_values))
    if not k_list:
        raise ValueError("k_values must not be empty")

    query_ids = sorted(qrels)
    if not query_ids:
        raise ValueError("qrels is empty")

    max_k = max(k_list)
    per_query: dict[str, dict[str, float]] = {}

    for query_id in query_ids:
        relevance = qrels[query_id]
        ranked = _ranked_doc_ids(run.get(query_id, {}), max_k)
        scores: dict[str, float] = {}
        for k in k_list:
            scores[f"ndcg@{k}"] = ndcg_at_k(ranked, relevance, k, gain=gain)
            scores[f"recall@{k}"] = recall_at_k(ranked, relevance, k)
        per_query[query_id] = scores

    return per_query


def evaluate_run(
    run: Run,
    qrels: Qrels,
    k_values: Iterable[int] = (10, 100),
    gain: GainFn = "exponential",
) -> dict[str, float]:
    """Score a full run, averaging each metric over queries.

    Queries present in ``qrels`` but absent from ``run`` are scored as 0 rather than
    dropped: a retriever that returns nothing for a query has failed that query, and
    silently excluding it inflates the mean.

    Args:
        run: ``{query_id: {doc_id: score}}``. Higher score means higher rank.
        qrels: ``{query_id: {doc_id: relevance_level}}``.
        k_values: Rank cutoffs to report.
        gain: Gain function for nDCG.

    Returns:
        ``{"ndcg@10": ..., "recall@10": ..., ...}`` plus ``"num_queries"``.
    """
    per_query = evaluate_run_per_query(run, qrels, k_values=k_values, gain=gain)

    n = len(per_query)
    totals: dict[str, float] = {}
    for scores in per_query.values():
        for name, value in scores.items():
            totals[name] = totals.get(name, 0.0) + value

    results = {name: total / n for name, total in totals.items()}
    results["num_queries"] = float(n)
    return results
