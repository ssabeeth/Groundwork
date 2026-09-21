"""Send each query to one retrieval system rather than combining them.

Experiment 8's finding is a correlation: BM25's per-query advantage over a dense encoder
rises with the rarity of the query's rarest term. A correlation is a statement about
measurement. Routing is the same statement made actionable — if the predictor really
tracks which system will win, then choosing per query on the predictor alone should beat
either system on its own.

Routing differs from fusion in what it throws away. Reciprocal rank fusion uses both
rankings for every query; a router commits to one and discards the other. That is the
whole reason to measure it rather than assume it: a router can only win by being right
about which system to pick, so its score is a direct readout of how much the predictor
knows.

An assignment maps each query id to the name of the system that will answer it. Two ways
of producing one live here:

- :func:`threshold_assignment` splits on a predictor computed before any retrieval runs,
  which is the honest kind and the kind that could be deployed.
- :func:`oracle_assignment` picks the system that actually scored better, using the
  relevance labels. It cannot be deployed and is not meant to be: it is the ceiling. No
  router driven by any predictor can beat it, so if the oracle is barely above the better
  single system, the question is closed for every predictor at once and not just for the
  one this project happens to have tried.
"""

from __future__ import annotations

from collections.abc import Mapping


def route_runs(
    runs: Mapping[str, Mapping[str, Mapping[str, float]]],
    assignment: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Build one run by taking each query's ranking from its assigned system.

    Args:
        runs: Retrieval runs by system name, each ``{query_id: {doc_id: score}}``.
        assignment: Which system answers each query, ``{query_id: system_name}``.

    Returns:
        A run containing exactly the queries in ``assignment``.

    Raises:
        KeyError: If an assignment names a system that was not supplied, or a query that
            the assigned system did not answer. Both are wiring errors, and a router that
            silently dropped queries would quietly inflate its own mean.
    """
    routed: dict[str, dict[str, float]] = {}
    for query_id, system in assignment.items():
        if system not in runs:
            raise KeyError(f"query {query_id} routed to unknown system {system!r}")
        run = runs[system]
        if query_id not in run:
            raise KeyError(f"system {system!r} has no ranking for query {query_id}")
        routed[query_id] = dict(run[query_id])
    return routed


def threshold_assignment(
    statistics: Mapping[str, Mapping[str, float]],
    predictor: str,
    threshold: float,
    at_or_above: str,
    below: str,
) -> dict[str, str]:
    """Route on one predictor, splitting at ``threshold``.

    The comparison is ``>=``, so ``threshold`` at or below the smallest observed value
    sends every query to ``at_or_above`` and recovers that system exactly. That degenerate
    end of the sweep is deliberate: it means the sweep always contains both single-system
    baselines, and a router that cannot beat them cannot hide behind a grid that excluded
    them.

    Args:
        statistics: Per-query predictor values, ``{query_id: {predictor: value}}``.
        predictor: Which statistic to split on.
        threshold: The split point.
        at_or_above: System answering queries whose predictor is ``>= threshold``.
        below: System answering the rest.

    Returns:
        ``{query_id: system_name}`` for every query in ``statistics``.
    """
    return {
        query_id: at_or_above if values[predictor] >= threshold else below
        for query_id, values in statistics.items()
    }


def oracle_assignment(
    per_query_scores: Mapping[str, Mapping[str, float]],
    metric: str,
) -> dict[str, str]:
    """Route each query to whichever system actually scored higher on it.

    This reads the relevance labels through ``per_query_scores`` and so is an upper
    bound, not a method. Ties go to the alphabetically first system name, which keeps the
    result independent of dictionary ordering; a tie contributes the same score either
    way, so the choice cannot affect the bound.

    Args:
        per_query_scores: ``{system_name: {query_id: score}}`` for one metric.
        metric: Name of the metric, used only in the error message.

    Returns:
        ``{query_id: system_name}``.

    Raises:
        ValueError: If the systems were not scored on identical query sets, which would
            make the bound an average over different queries for different systems.
    """
    systems = sorted(per_query_scores)
    if not systems:
        raise ValueError("no systems to route between")
    query_sets = [set(per_query_scores[name]) for name in systems]
    if any(other != query_sets[0] for other in query_sets[1:]):
        raise ValueError(f"systems were scored on different query sets for {metric}")

    # ``systems`` is sorted and ``max`` returns the first maximal element, so ties fall to
    # the alphabetically first name without needing a second sort key.
    return {
        query_id: max(systems, key=lambda name: per_query_scores[name][query_id])
        for query_id in query_sets[0]
    }
