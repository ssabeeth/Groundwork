"""Routing: each query answered by one system, chosen by a rule.

Expected values here are worked out from the definitions in the module docstring, not
copied from what the functions returned. The oracle bound is the number that decides
whether query routing is worth anything at all, so an oracle that quietly picked the
wrong system, or averaged over a different query set for each system, would retire a
whole line of enquiry for the wrong reason.
"""

import pytest

from groundwork.retrieval.routing import (
    oracle_assignment,
    route_runs,
    threshold_assignment,
)

RUNS = {
    "lexical": {
        "q1": {"d1": 3.0, "d2": 1.0},
        "q2": {"d3": 2.0},
        "q3": {"d4": 5.0},
    },
    "semantic": {
        "q1": {"d9": 0.9},
        "q2": {"d8": 0.8, "d7": 0.7},
        "q3": {"d6": 0.6},
    },
}

# max_idf per query. The values are chosen so that a threshold of 4.0 splits them
# 2-versus-1 and a threshold of 2.0 sends every query to the same side.
STATISTICS = {
    "q1": {"max_idf": 6.0, "num_terms": 3.0},
    "q2": {"max_idf": 4.0, "num_terms": 7.0},
    "q3": {"max_idf": 2.0, "num_terms": 5.0},
}


def test_routed_run_takes_each_querys_ranking_from_its_assigned_system():
    routed = route_runs(RUNS, {"q1": "lexical", "q2": "semantic", "q3": "lexical"})
    assert routed == {
        "q1": {"d1": 3.0, "d2": 1.0},
        "q2": {"d8": 0.8, "d7": 0.7},
        "q3": {"d4": 5.0},
    }


def test_routed_run_contains_exactly_the_assigned_queries():
    """A router that dropped queries would inflate its own mean, since the scorer
    averages over the queries present."""
    routed = route_runs(RUNS, {"q1": "lexical"})
    assert set(routed) == {"q1"}


def test_routing_to_an_unknown_system_fails_loudly():
    with pytest.raises(KeyError, match="unknown system"):
        route_runs(RUNS, {"q1": "hybrid"})


def test_routing_a_query_the_assigned_system_never_answered_fails_loudly():
    with pytest.raises(KeyError, match="no ranking for query"):
        route_runs({"lexical": {"q1": {"d1": 1.0}}}, {"q2": "lexical"})


def test_routed_run_is_a_copy_so_editing_it_cannot_corrupt_the_source_run():
    routed = route_runs(RUNS, {"q1": "lexical"})
    routed["q1"]["d1"] = 99.0
    assert RUNS["lexical"]["q1"]["d1"] == 3.0


def test_threshold_splits_at_or_above_from_below():
    # max_idf: q1 6.0, q2 4.0, q3 2.0. At threshold 4.0 the comparison is >=, so q1 and
    # q2 go above and q3 alone goes below.
    assignment = threshold_assignment(STATISTICS, "max_idf", 4.0, "lexical", "semantic")
    assert assignment == {"q1": "lexical", "q2": "lexical", "q3": "semantic"}


def test_a_threshold_at_or_below_the_smallest_value_recovers_one_system_exactly():
    """The degenerate ends of the sweep must be reachable, so that a router is always
    compared against a grid containing both single-system baselines."""
    assignment = threshold_assignment(STATISTICS, "max_idf", 2.0, "lexical", "semantic")
    assert set(assignment.values()) == {"lexical"}


def test_a_threshold_above_every_value_recovers_the_other_system_exactly():
    assignment = threshold_assignment(STATISTICS, "max_idf", 6.5, "lexical", "semantic")
    assert set(assignment.values()) == {"semantic"}


def test_threshold_can_split_on_a_predictor_other_than_the_primary_one():
    # num_terms: q1 3.0, q2 7.0, q3 5.0. At 5.0, q2 and q3 are at or above.
    assignment = threshold_assignment(STATISTICS, "num_terms", 5.0, "long", "short")
    assert assignment == {"q1": "short", "q2": "long", "q3": "long"}


def test_oracle_picks_the_higher_scoring_system_per_query():
    scores = {
        "lexical": {"q1": 0.9, "q2": 0.1, "q3": 0.5},
        "semantic": {"q1": 0.2, "q2": 0.8, "q3": 0.4},
    }
    assert oracle_assignment(scores, "ndcg@10") == {
        "q1": "lexical",
        "q2": "semantic",
        "q3": "lexical",
    }


def test_oracle_breaks_ties_toward_the_alphabetically_first_system():
    """Either choice gives the same bound, so the rule exists only to make the output
    independent of dictionary ordering."""
    scores = {"semantic": {"q1": 0.5}, "lexical": {"q1": 0.5}}
    assert oracle_assignment(scores, "ndcg@10") == {"q1": "lexical"}


def test_oracle_mean_equals_the_mean_of_per_query_maxima():
    """The definition of the bound: routing perfectly is taking the better score every
    time, so the oracle's mean is the mean of the per-query maxima.

    Worked by hand: max(0.9, 0.2) + max(0.1, 0.8) + max(0.5, 0.4) = 0.9 + 0.8 + 0.5,
    over three queries, is 2.2 / 3.
    """
    scores = {
        "lexical": {"q1": 0.9, "q2": 0.1, "q3": 0.5},
        "semantic": {"q1": 0.2, "q2": 0.8, "q3": 0.4},
    }
    assignment = oracle_assignment(scores, "ndcg@10")
    routed_mean = sum(scores[system][qid] for qid, system in assignment.items()) / len(assignment)
    assert routed_mean == pytest.approx(2.2 / 3)


def test_oracle_is_never_below_either_system_it_routes_between():
    """Structural property of a maximum, and the reason the bound is a bound."""
    scores = {
        "lexical": {"q1": 0.9, "q2": 0.1, "q3": 0.5},
        "semantic": {"q1": 0.2, "q2": 0.8, "q3": 0.4},
    }
    assignment = oracle_assignment(scores, "ndcg@10")
    routed = [scores[system][qid] for qid, system in assignment.items()]
    for name, per_query in scores.items():
        assert sum(routed) >= sum(per_query.values()), f"oracle fell below {name}"


def test_oracle_refuses_systems_scored_on_different_query_sets():
    """Averaging one system over three queries and the other over two would compare
    means that are not comparable, and the bound would be meaningless."""
    scores = {"lexical": {"q1": 0.9, "q2": 0.1}, "semantic": {"q1": 0.2}}
    with pytest.raises(ValueError, match="different query sets"):
        oracle_assignment(scores, "ndcg@10")


def test_oracle_with_no_systems_fails_rather_than_returning_an_empty_assignment():
    with pytest.raises(ValueError, match="no systems"):
        oracle_assignment({}, "ndcg@10")
