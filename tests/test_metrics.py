"""Metric tests against values worked out by hand from the definitions.

The expected numbers below are literals, derived on paper and shown in the comments.
They are deliberately not recomputed with the same formula the implementation uses,
because a test that mirrors the implementation only proves the code is consistent
with itself.

Useful constants:
    1 / log2(2) = 1.0
    1 / log2(3) = 0.6309297535714574
    1 / log2(4) = 0.5
    1 / log2(5) = 0.43067655807339306
    1 / log2(6) = 0.3868528072345416
"""

import pytest

from groundwork.eval.metrics import (
    evaluate_run,
    evaluate_run_per_query,
    ndcg_at_k,
    oracle_recall_at_k,
    recall_at_k,
)


class TestNDCG:
    def test_perfect_ranking_scores_one(self):
        # Both relevant docs sit at ranks 1 and 2, so DCG == IDCG.
        ranked = ["d1", "d2", "d3"]
        relevance = {"d1": 1, "d2": 1}
        assert ndcg_at_k(ranked, relevance, k=3) == pytest.approx(1.0)

    def test_single_relevant_doc_at_rank_two(self):
        # DCG = 1/log2(3) = 0.6309297535714574; IDCG = 1/log2(2) = 1.0
        ranked = ["d9", "d1", "d8"]
        relevance = {"d1": 1}
        assert ndcg_at_k(ranked, relevance, k=10) == pytest.approx(0.6309297535714574)

    def test_reversed_ranking_with_graded_relevance_exponential(self):
        # relevance a=2, b=1 -> exponential gains 3 and 1.
        # DCG  = 1/log2(2) + 3/log2(3) = 1 + 1.8927892607143721 = 2.892789260714372
        # IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309297535714574 = 3.6309297535714573
        # nDCG = 2.8927892607143724 / 3.6309297535714578 = 0.7967075809905066
        ranked = ["b", "a"]
        relevance = {"a": 2, "b": 1}
        assert ndcg_at_k(ranked, relevance, k=10, gain="exponential") == pytest.approx(
            0.7967075809905066, abs=1e-12
        )

    def test_reversed_ranking_with_graded_relevance_linear(self):
        # Same ranking, linear gains 2 and 1.
        # DCG  = 1 + 2/log2(3) = 2.2618595071429148
        # IDCG = 2 + 1/log2(3) = 2.6309297535714573
        # nDCG = 0.8597187...
        ranked = ["b", "a"]
        relevance = {"a": 2, "b": 1}
        assert ndcg_at_k(ranked, relevance, k=10, gain="linear") == pytest.approx(
            0.8597187, abs=1e-7
        )

    def test_gain_functions_agree_on_binary_qrels(self):
        # 2**1 - 1 == 1, so the two gain functions must coincide exactly when every
        # judgement is 0 or 1. This is why SciFact can be scored either way.
        ranked = ["d3", "d1", "d5", "d2"]
        relevance = {"d1": 1, "d2": 1, "d4": 1}
        exponential = ndcg_at_k(ranked, relevance, k=4, gain="exponential")
        linear = ndcg_at_k(ranked, relevance, k=4, gain="linear")
        assert exponential == pytest.approx(linear)

    def test_idcg_counts_relevant_docs_that_were_never_retrieved(self):
        # Five relevant docs exist; the system found one, at rank 1.
        # DCG  = 1.0
        # IDCG = 1 + 0.6309297535714574 + 0.5 + 0.43067655807339306 + 0.3868528072345416
        #      = 2.948459118879392
        # nDCG = 0.3391602...
        ranked = ["d1", "x", "y", "z", "w"]
        relevance = {"d1": 1, "d2": 1, "d3": 1, "d4": 1, "d5": 1}
        assert ndcg_at_k(ranked, relevance, k=5) == pytest.approx(0.3391602, abs=1e-7)

    def test_cutoff_truncates_the_ideal_ranking_too(self):
        # k=1 with two relevant docs: IDCG is the single best document, not both.
        # Relevant doc at rank 1 -> DCG = 1.0, IDCG = 1.0, nDCG = 1.0
        ranked = ["d1", "d2"]
        relevance = {"d1": 1, "d2": 1}
        assert ndcg_at_k(ranked, relevance, k=1) == pytest.approx(1.0)

    def test_no_relevant_documents_scores_zero(self):
        assert ndcg_at_k(["a", "b"], {"c": 0}, k=10) == 0.0

    def test_unjudged_documents_count_as_non_relevant(self):
        # "unknown" is absent from qrels and must not contribute gain.
        with_unjudged = ndcg_at_k(["unknown", "d1"], {"d1": 1}, k=10)
        assert with_unjudged == pytest.approx(0.6309297535714574)

    def test_promoting_a_relevant_document_cannot_hurt(self):
        relevance = {"d1": 1, "d2": 2}
        worse = ndcg_at_k(["x", "d1", "y", "d2"], relevance, k=10)
        better = ndcg_at_k(["d2", "d1", "x", "y"], relevance, k=10)
        assert better > worse

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], {"a": 1}, k=0)


class TestRecall:
    def test_counts_relevant_documents_within_cutoff(self):
        ranked = ["d1", "x", "d2"]
        relevance = {"d1": 1, "d2": 1, "d3": 1}
        assert recall_at_k(ranked, relevance, k=3) == pytest.approx(2 / 3)

    def test_cutoff_is_respected(self):
        ranked = ["d1", "x", "d2"]
        relevance = {"d1": 1, "d2": 1, "d3": 1}
        assert recall_at_k(ranked, relevance, k=1) == pytest.approx(1 / 3)

    def test_graded_levels_all_count_as_relevant(self):
        ranked = ["d1", "d2"]
        relevance = {"d1": 2, "d2": 1}
        assert recall_at_k(ranked, relevance, k=2) == pytest.approx(1.0)

    def test_zero_relevance_level_is_not_relevant(self):
        assert recall_at_k(["d1"], {"d1": 0}, k=10) == 0.0

    def test_no_relevant_documents_scores_zero(self):
        assert recall_at_k(["a"], {"b": 0}, k=10) == 0.0


class TestEvaluateRun:
    def test_averages_over_queries(self):
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        run = {"q1": {"d1": 5.0}, "q2": {"d2": 5.0}}
        results = evaluate_run(run, qrels, k_values=[10])
        assert results["ndcg@10"] == pytest.approx(1.0)
        assert results["recall@10"] == pytest.approx(1.0)
        assert results["num_queries"] == 2.0

    def test_query_missing_from_run_scores_zero_rather_than_being_dropped(self):
        # q1 is perfect, q2 returned nothing. The mean must be 0.5, not 1.0.
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        run = {"q1": {"d1": 5.0, "d2": 1.0}}
        results = evaluate_run(run, qrels, k_values=[10])
        assert results["ndcg@10"] == pytest.approx(0.5)
        assert results["recall@10"] == pytest.approx(0.5)
        assert results["num_queries"] == 2.0

    def test_ties_are_broken_by_document_id(self):
        # b and a tie on score 1.0; c wins on score. Order must be c, a, b,
        # putting the relevant doc "a" at rank 2 -> 1/log2(3).
        qrels = {"q1": {"a": 1}}
        run = {"q1": {"b": 1.0, "a": 1.0, "c": 2.0}}
        results = evaluate_run(run, qrels, k_values=[3])
        assert results["ndcg@3"] == pytest.approx(0.6309297535714574, abs=1e-7)

    def test_reports_every_requested_cutoff(self):
        qrels = {"q1": {"d1": 1}}
        run = {"q1": {"d1": 1.0}}
        results = evaluate_run(run, qrels, k_values=[1, 10, 100])
        assert set(results) == {
            "ndcg@1",
            "ndcg@10",
            "ndcg@100",
            "recall@1",
            "recall@10",
            "recall@100",
            "num_queries",
        }

    def test_rejects_empty_qrels(self):
        with pytest.raises(ValueError):
            evaluate_run({}, {}, k_values=[10])


class TestEvaluateRunPerQuery:
    def test_scores_each_query_separately_with_hand_computed_values(self):
        # q1: d1 is relevant and sits at rank 1 -> DCG = IDCG = 1.0, nDCG = 1.0
        # q2: d2 is relevant and sits at rank 2 -> DCG = 1/log2(3), IDCG = 1.0
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        run = {"q1": {"d1": 5.0, "x": 1.0}, "q2": {"y": 1.0, "d2": 0.5}}
        per_query = evaluate_run_per_query(run, qrels, k_values=[10])

        assert per_query["q1"]["ndcg@10"] == pytest.approx(1.0)
        assert per_query["q2"]["ndcg@10"] == pytest.approx(0.6309297535714574)
        assert per_query["q1"]["recall@10"] == pytest.approx(1.0)
        assert per_query["q2"]["recall@10"] == pytest.approx(1.0)

    def test_one_entry_per_judged_query_ordered_by_id(self):
        qrels = {"q2": {"d1": 1}, "q1": {"d1": 1}, "q10": {"d1": 1}}
        run = {"q1": {"d1": 1.0}}
        per_query = evaluate_run_per_query(run, qrels, k_values=[10])
        assert list(per_query) == ["q1", "q10", "q2"]

    def test_query_missing_from_the_run_scores_zero(self):
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        run = {"q1": {"d1": 1.0}}
        per_query = evaluate_run_per_query(run, qrels, k_values=[10])
        assert per_query["q2"]["ndcg@10"] == 0.0
        assert per_query["q2"]["recall@10"] == 0.0

    def test_reports_every_requested_cutoff(self):
        qrels = {"q1": {"d1": 1}}
        run = {"q1": {"d1": 1.0}}
        per_query = evaluate_run_per_query(run, qrels, k_values=[1, 10])
        assert set(per_query["q1"]) == {"ndcg@1", "recall@1", "ndcg@10", "recall@10"}

    def test_rejects_empty_qrels(self):
        with pytest.raises(ValueError):
            evaluate_run_per_query({}, {}, k_values=[10])


class TestOracleRecall:
    def test_one_relevant_document_per_query_allows_perfect_recall(self):
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        assert oracle_recall_at_k(qrels, k=100) == pytest.approx(1.0)

    def test_more_relevant_documents_than_the_cutoff_caps_recall(self):
        # 4 relevant documents, k=2 -> the best possible is 2/4 = 0.5.
        qrels = {"q1": {"a": 1, "b": 1, "c": 1, "d": 1}}
        assert oracle_recall_at_k(qrels, k=2) == pytest.approx(0.5)

    def test_averages_over_queries_with_different_counts(self):
        # q1: 1 relevant, k=2 -> min(2,1)/1 = 1.0
        # q2: 4 relevant, k=2 -> min(2,4)/4 = 0.5
        # mean = 0.75
        qrels = {"q1": {"a": 1}, "q2": {"w": 1, "x": 1, "y": 1, "z": 1}}
        assert oracle_recall_at_k(qrels, k=2) == pytest.approx(0.75)

    def test_graded_levels_all_count_as_relevant(self):
        qrels = {"q1": {"a": 2, "b": 1}}
        assert oracle_recall_at_k(qrels, k=2) == pytest.approx(1.0)

    def test_zero_level_judgements_are_not_relevant(self):
        # Only "a" is relevant, so k=1 already reaches the ceiling.
        qrels = {"q1": {"a": 1, "b": 0, "c": 0}}
        assert oracle_recall_at_k(qrels, k=1) == pytest.approx(1.0)

    def test_a_query_with_no_relevant_documents_scores_zero(self):
        qrels = {"q1": {"a": 0}, "q2": {"b": 1}}
        assert oracle_recall_at_k(qrels, k=10) == pytest.approx(0.5)

    def test_is_an_upper_bound_on_any_actual_run(self):
        qrels = {"q1": {"a": 1, "b": 1, "c": 1}, "q2": {"d": 1}}
        run = {"q1": {"a": 3.0, "b": 2.0}, "q2": {"d": 1.0}}
        measured = evaluate_run(run, qrels, k_values=[2])["recall@2"]
        assert measured <= oracle_recall_at_k(qrels, k=2) + 1e-12

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            oracle_recall_at_k({"q1": {"a": 1}}, k=0)

    def test_rejects_empty_qrels(self):
        with pytest.raises(ValueError):
            oracle_recall_at_k({}, k=10)
