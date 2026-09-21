"""Chance-corrected agreement between two assessors.

Every expected value below is worked out by hand from

    kappa = (p_observed - p_chance) / (1 - p_chance)

with the arithmetic shown, because this is the number that decides whether an LLM judge
is trustworthy enough to evaluate anything. A kappa implementation that agreed with
itself would let a judge that answers "not relevant" to everything look like a good one.
"""

import pytest

from groundwork.eval.agreement import cohens_kappa, confusion_matrix


class TestConfusionMatrix:
    def test_counts_every_label_pair(self):
        labels, matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
        assert labels == [0, 1]
        # first=0,second=0 once; first=0,second=1 once; first=1,second=1 twice.
        assert matrix.tolist() == [[1, 1], [0, 2]]

    def test_labels_absent_from_one_assessor_still_appear(self):
        labels, matrix = confusion_matrix([0, 0], [0, 2])
        assert labels == [0, 2]
        assert matrix.tolist() == [[1, 1], [0, 0]]

    def test_mismatched_lengths_fail_loudly(self):
        """Silently zipping to the shorter sequence would align different items."""
        with pytest.raises(ValueError, match="different numbers of items"):
            confusion_matrix([0, 1, 1], [0, 1])


class TestCohensKappa:
    def test_perfect_agreement_is_one(self):
        result = cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1])
        assert result.cohens_kappa == pytest.approx(1.0)
        assert result.raw_agreement == pytest.approx(1.0)

    def test_worked_example(self):
        """Ten items. Assessor A: six 1s, four 0s. Assessor B: five 1s, five 0s.
        They agree on five 1s and four 0s... which is nine of ten.

        A = [1,1,1,1,1,1,0,0,0,0]
        B = [1,1,1,1,1,0,0,0,0,0]

        observed = 9/10 = 0.9
        A says 1 six times (0.6), B says 1 five times (0.5)
        A says 0 four times (0.4), B says 0 five times (0.5)
        chance = 0.6*0.5 + 0.4*0.5 = 0.30 + 0.20 = 0.5
        kappa = (0.9 - 0.5) / (1 - 0.5) = 0.4 / 0.5 = 0.8
        """
        a = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        b = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        result = cohens_kappa(a, b)
        assert result.raw_agreement == pytest.approx(0.9)
        assert result.chance_agreement == pytest.approx(0.5)
        assert result.cohens_kappa == pytest.approx(0.8)

    def test_chance_level_agreement_is_zero(self):
        """Four items covering every combination once. Each assessor says 1 half the
        time, so chance = 0.5*0.5 + 0.5*0.5 = 0.5, and they agree on two of four = 0.5.
        kappa = (0.5 - 0.5) / 0.5 = 0.
        """
        result = cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1])
        assert result.raw_agreement == pytest.approx(0.5)
        assert result.chance_agreement == pytest.approx(0.5)
        assert result.cohens_kappa == pytest.approx(0.0)

    def test_systematic_disagreement_is_negative(self):
        """Perfect inversion. observed = 0; each says 1 half the time so chance = 0.5;
        kappa = (0 - 0.5) / 0.5 = -1."""
        result = cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0])
        assert result.cohens_kappa == pytest.approx(-1.0)

    def test_a_judge_that_always_says_no_scores_zero_despite_high_raw_agreement(self):
        """The failure this metric exists to catch.

        Twenty items, one of them relevant. A judge answering "not relevant" to all
        twenty matches the human on nineteen — 95% raw agreement — while having
        discriminated nothing. Chance agreement is also 0.95, so kappa is 0.
        """
        human = [1] + [0] * 19
        judge = [0] * 20
        result = cohens_kappa(human, judge)
        assert result.raw_agreement == pytest.approx(0.95)
        assert result.cohens_kappa == pytest.approx(0.0)

    def test_both_assessors_constant_gives_zero_rather_than_dividing_by_zero(self):
        """Chance agreement is exactly 1.0 here, so the formula is undefined. Zero is
        the honest reading: two assessors who only ever say one thing have shown no
        agreement beyond chance."""
        result = cohens_kappa([0, 0, 0], [0, 0, 0])
        assert result.chance_agreement == pytest.approx(1.0)
        assert result.cohens_kappa == pytest.approx(0.0)

    def test_kappa_is_symmetric_in_its_two_assessors(self):
        a = [0, 1, 1, 0, 1, 0, 0, 1]
        b = [0, 1, 0, 0, 1, 1, 0, 1]
        assert cohens_kappa(a, b).cohens_kappa == pytest.approx(cohens_kappa(b, a).cohens_kappa)

    def test_graded_labels_are_supported(self):
        """BEIR qrels carry grades 0, 1 and 2, so a judge asked for a grade must be
        scorable against them without collapsing to binary first."""
        result = cohens_kappa([0, 1, 2, 2, 1], [0, 1, 2, 1, 1])
        assert result.labels == [0, 1, 2]
        assert result.raw_agreement == pytest.approx(0.8)

    def test_no_items_fails_loudly(self):
        with pytest.raises(ValueError, match="no items"):
            cohens_kappa([], [])

    def test_describe_is_json_safe(self):
        import json

        described = cohens_kappa([0, 1, 1, 0], [0, 1, 0, 0]).describe()
        json.dumps(described)
        assert described["test"] == "cohens_kappa"
        assert described["num_items"] == 4
