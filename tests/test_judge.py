"""The LLM relevance judge and its calibration against human assessors.

The model is not loaded here — CI runs offline. What is tested is the surrounding logic,
which is where a judge evaluation goes wrong quietly: which pairs are eligible to be
scored, how an unparseable answer is handled, and the refusal to calibrate on a pool with
no negative human labels.

Every one of those, done wrong, produces a kappa that looks fine and means nothing.
"""

import pytest

from groundwork.eval.judge import calibrate, judgeable_pairs, parse_verdict

# Grades include explicit zeros, as SciDocs and TREC-COVID do.
QRELS = {
    "q1": {"d1": 1, "d2": 0, "d3": 0},
    "q2": {"d4": 2, "d5": 0},
}
RUN = {
    "q1": {"d1": 3.0, "d2": 2.0, "d9": 1.0},
    "q2": {"d5": 5.0, "d4": 4.0},
}
# Four relevant and four not, for testing the qrels source without tripping the
# class-balance guard.
BALANCED = {
    "q1": {"a1": 1, "a2": 1, "a3": 0, "a4": 0},
    "q2": {"b1": 2, "b2": 1, "b3": 0, "b4": 0},
}


class TestParseVerdict:
    @pytest.mark.parametrize("text", ["yes", "Yes", "  YES  ", "true", "relevant"])
    def test_affirmative_answers(self, text):
        assert parse_verdict(text) == 1

    @pytest.mark.parametrize("text", ["no", "No", "false", "irrelevant"])
    def test_negative_answers(self, text):
        assert parse_verdict(text) == 0

    def test_not_relevant_is_negative_despite_containing_relevant(self):
        """'not relevant' starts with 'not' and contains 'relevant'. Checking
        affirmatives first would classify it as a yes."""
        assert parse_verdict("not relevant") == 0

    @pytest.mark.parametrize("text", ["", "   ", "maybe", "the document discusses"])
    def test_unparseable_answers_return_none_rather_than_a_class(self, text):
        """Coercing an unparseable answer to 'no' would inflate agreement, because the
        pool is mostly non-relevant and a broken judge would look conservative."""
        assert parse_verdict(text) is None


class TestJudgeablePairs:
    """Pool construction, which is where this experiment first went wrong.

    The original version sampled the top 10 of a retrieval run, on the reasoning that
    those are the documents a RAG system would show a model. On SciDocs that produced 819
    labelled pairs of which 814 were relevant, and a kappa of 0.011 that reflected the
    pool rather than the judge.
    """

    def test_qrels_source_takes_every_human_labelled_pair(self):
        pairs = judgeable_pairs(BALANCED, min_per_class=2)
        assert len(pairs) == 8

    def test_retrieved_source_is_limited_to_the_ranking(self):
        pairs = judgeable_pairs(QRELS, RUN, depth=10, source="retrieved", min_per_class=1)
        assert {(q, d) for q, d, _ in pairs} == {
            ("q1", "d1"),
            ("q1", "d2"),
            ("q2", "d4"),
            ("q2", "d5"),
        }

    def test_an_unjudged_retrieved_document_is_not_eligible(self):
        """d9 was retrieved but never judged, so there is nothing to agree with."""
        pairs = judgeable_pairs(QRELS, RUN, depth=10, source="retrieved", min_per_class=1)
        assert ("q1", "d9", 0) not in pairs

    def test_the_human_grade_travels_with_the_pair(self):
        pairs = {
            (q, d): g
            for q, d, g in judgeable_pairs(
                QRELS, RUN, depth=10, source="retrieved", min_per_class=1
            )
        }
        assert pairs[("q1", "d1")] == 1
        assert pairs[("q2", "d4")] == 2
        assert pairs[("q1", "d2")] == 0

    def test_depth_limits_how_far_down_each_ranking_it_looks(self):
        pairs = judgeable_pairs(QRELS, RUN, depth=1, source="retrieved", min_per_class=0)
        assert {(q, d) for q, d, _ in pairs} == {("q1", "d1"), ("q2", "d5")}

    def test_qrels_without_negative_labels_are_refused(self):
        """SciFact and NFCorpus ship only positive judgements."""
        positive_only = {"q1": {"d1": 1, "d2": 1}, "q2": {"d4": 2}}
        with pytest.raises(ValueError, match="no non-relevant judgements"):
            judgeable_pairs(positive_only, min_per_class=0)

    def test_a_pool_that_is_almost_all_one_class_is_refused(self):
        """The failure that produced an uninterpretable kappa on the first run. Chance
        agreement approaches the observed rate, so kappa approaches zero however good
        the judge is."""
        lopsided = {"q1": {f"d{i}": 1 for i in range(50)} | {"d99": 0}}
        with pytest.raises(ValueError, match="at least 30 of each"):
            judgeable_pairs(lopsided)

    def test_retrieved_without_a_run_fails_loudly(self):
        with pytest.raises(ValueError, match="needs a run"):
            judgeable_pairs(QRELS, source="retrieved", min_per_class=0)

    def test_an_unknown_source_fails_loudly(self):
        with pytest.raises(ValueError, match='must be "qrels" or "retrieved"'):
            judgeable_pairs(QRELS, source="sampled", min_per_class=0)


class TestCalibrate:
    def test_unparseable_verdicts_are_dropped_and_counted(self):
        agreement, unparseable = calibrate([1, 0, 1, 0], [1, 0, None, 0])
        assert unparseable == 1
        assert agreement.num_items == 3

    def test_graded_human_labels_collapse_to_binary(self):
        """The judge is asked yes/no, so grade 2 and grade 1 are both 'relevant'."""
        agreement, _ = calibrate([2, 1, 0], [1, 1, 0])
        assert agreement.raw_agreement == pytest.approx(1.0)

    def test_a_judge_that_always_says_yes_scores_zero_kappa(self):
        """The mirror of the always-no failure, and the one an eager judge produces."""
        human = [1, 0, 0, 0, 0]
        verdicts = [1, 1, 1, 1, 1]
        agreement, _ = calibrate(human, verdicts)
        assert agreement.cohens_kappa == pytest.approx(0.0)

    def test_mismatched_lengths_fail_loudly(self):
        with pytest.raises(ValueError, match="human labels against"):
            calibrate([1, 0, 1], [1, 0])

    def test_all_unparseable_fails_rather_than_reporting_a_kappa(self):
        """Returning a kappa over zero items would be a number with no data behind it."""
        with pytest.raises(ValueError, match="no parseable verdicts"):
            calibrate([1, 0], [None, None])
