"""Reciprocal rank fusion tests.

Expected values are worked out by hand from the definition, which is small enough to do
exactly: RRF(d) = sum over systems of 1/(k + rank(d)), ranks starting at 1.
"""

import pytest

from groundwork.eval import evaluate_run
from groundwork.retrieval.fusion import reciprocal_rank_fusion


class TestArithmetic:
    def test_single_document_single_run(self):
        # One system, one document at rank 1, k=60 -> 1/61.
        fused = reciprocal_rank_fusion([{"q1": {"a": 9.9}}], k=60.0)
        assert fused["q1"]["a"] == pytest.approx(1 / 61)

    def test_scores_are_discarded_and_only_ranks_count(self):
        # Wildly different scores, identical ranks -> identical fusion.
        big = {"q1": {"a": 1000.0, "b": 999.0}}
        small = {"q1": {"a": 0.002, "b": 0.001}}
        assert reciprocal_rank_fusion([big])["q1"] == pytest.approx(
            reciprocal_rank_fusion([small])["q1"]
        )

    def test_two_runs_agreeing_sum_their_contributions(self):
        # "a" is rank 1 in both -> 2/61. "b" is rank 2 in both -> 2/62.
        run = {"q1": {"a": 2.0, "b": 1.0}}
        fused = reciprocal_rank_fusion([run, run], k=60.0)
        assert fused["q1"]["a"] == pytest.approx(2 / 61)
        assert fused["q1"]["b"] == pytest.approx(2 / 62)

    def test_two_runs_disagreeing(self):
        # a: rank 1 then rank 2 -> 1/61 + 1/62 = 0.032523...
        # b: rank 2 then rank 1 -> 1/62 + 1/61, identical, so they tie and sort by id.
        first = {"q1": {"a": 2.0, "b": 1.0}}
        second = {"q1": {"b": 2.0, "a": 1.0}}
        fused = reciprocal_rank_fusion([first, second], k=60.0)
        assert fused["q1"]["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["q1"]["a"] == pytest.approx(fused["q1"]["b"])

    def test_a_document_only_one_system_found_still_scores(self):
        # "c" appears once at rank 1 -> 1/61, from that system alone.
        fused = reciprocal_rank_fusion([{"q1": {"a": 2.0}}, {"q1": {"c": 5.0}}], k=60.0)
        assert fused["q1"]["c"] == pytest.approx(1 / 61)
        assert set(fused["q1"]) == {"a", "c"}

    def test_k_zero_makes_rank_one_dominate(self):
        # k=0 -> contributions 1/1, 1/2, 1/3: the steepest possible decay.
        fused = reciprocal_rank_fusion([{"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}], k=0.0)
        assert fused["q1"]["a"] == pytest.approx(1.0)
        assert fused["q1"]["b"] == pytest.approx(0.5)
        assert fused["q1"]["c"] == pytest.approx(1 / 3)

    def test_large_k_flattens_the_difference_between_ranks(self):
        steep = reciprocal_rank_fusion([{"q1": {"a": 2.0, "b": 1.0}}], k=1.0)["q1"]
        flat = reciprocal_rank_fusion([{"q1": {"a": 2.0, "b": 1.0}}], k=1000.0)["q1"]
        assert steep["a"] / steep["b"] > flat["a"] / flat["b"]

    def test_weights_scale_a_systems_contribution(self):
        run = {"q1": {"a": 1.0}}
        fused = reciprocal_rank_fusion([run, run], k=60.0, weights=[1.0, 3.0])
        assert fused["q1"]["a"] == pytest.approx(4 / 61)

    def test_zero_weight_removes_a_system(self):
        fused = reciprocal_rank_fusion([{"q1": {"a": 1.0}}, {"q1": {"b": 1.0}}], weights=[1.0, 0.0])
        assert set(fused["q1"]) == {"a"}


class TestBehaviour:
    def test_ties_are_broken_by_document_id(self):
        fused = reciprocal_rank_fusion([{"q1": {"a": 1.0, "b": 1.0, "c": 1.0}}])
        assert list(fused["q1"]) == ["a", "b", "c"]

    def test_top_k_is_respected(self):
        run = {"q1": {chr(97 + i): float(10 - i) for i in range(10)}}
        assert len(reciprocal_rank_fusion([run], top_k=3)["q1"]) == 3

    def test_queries_missing_from_one_run_still_appear(self):
        fused = reciprocal_rank_fusion([{"q1": {"a": 1.0}}, {"q2": {"b": 1.0}}])
        assert set(fused) == {"q1", "q2"}

    def test_an_empty_query_result_contributes_nothing(self):
        fused = reciprocal_rank_fusion([{"q1": {}}, {"q1": {"a": 1.0}}])
        assert fused["q1"] == pytest.approx({"a": 1 / 61})

    def test_fusing_one_run_preserves_its_ordering(self):
        # RRF is monotone in rank, so a single run comes back in the same order.
        run = {"q1": {"a": 5.0, "b": 3.0, "c": 1.0}}
        assert list(reciprocal_rank_fusion([run])["q1"]) == ["a", "b", "c"]

    def test_fusion_output_scores_like_any_other_run(self):
        # The point of the shared run shape: the scorer cannot tell fusion from anything.
        fused = reciprocal_rank_fusion([{"q1": {"a": 2.0, "b": 1.0}}])
        metrics = evaluate_run(fused, {"q1": {"a": 1}}, k_values=(10,))
        assert metrics["ndcg@10"] == pytest.approx(1.0)


class TestValidation:
    def test_no_runs_raises(self):
        with pytest.raises(ValueError, match="at least one run"):
            reciprocal_rank_fusion([])

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="k must be non-negative"):
            reciprocal_rank_fusion([{"q1": {"a": 1.0}}], k=-1.0)

    def test_non_positive_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k must be positive"):
            reciprocal_rank_fusion([{"q1": {"a": 1.0}}], top_k=0)

    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError, match="weights for"):
            reciprocal_rank_fusion([{"q1": {"a": 1.0}}], weights=[1.0, 1.0])
