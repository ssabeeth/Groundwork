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
    def test_only_pairs_a_human_labelled_are_eligible(self):
        """d9 was retrieved but never judged, so there is nothing to agree with."""
        pairs = judgeable_pairs(QRELS, RUN, depth=10)
        assert ("q1", "d9", 0) not in pairs
        assert {(q, d) for q, d, _ in pairs} == {
            ("q1", "d1"),
            ("q1", "d2"),
            ("q2", "d4"),
            ("q2", "d5"),
        }

    def test_the_human_grade_travels_with_the_pair(self):
        pairs = dict(((q, d), g) for q, d, g in judgeable_pairs(QRELS, RUN, depth=10))
        assert pairs[("q1", "d1")] == 1
        assert pairs[("q2", "d4")] == 2
        assert pairs[("q1", "d2")] == 0

    def test_depth_limits_how_far_down_each_ranking_it_looks(self):
        """d2 ranks second for q1, so depth=1 must exclude it."""
        pairs = judgeable_pairs(QRELS, RUN, depth=1)
        assert {(q, d) for q, d, _ in pairs} == {("q1", "d1"), ("q2", "d5")}

    def test_qrels_without_negative_labels_are_refused(self):
        """SciFact and NFCorpus ship only positive judgements. Calibrating on them would
        measure how often the judge says yes, not whether it agrees with anyone."""
        positive_only = {"q1": {"d1": 1, "d2": 1}, "q2": {"d4": 2}}
        with pytest.raises(ValueError, match="no non-relevant judgements"):
            judgeable_pairs(positive_only, RUN, depth=10)


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
