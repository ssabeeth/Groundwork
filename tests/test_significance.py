"""Significance test tests.

Same discipline as tests/test_bm25.py: the expected values here are either worked out
by hand from the definition of the randomisation test, with the enumeration shown in
the comments, or produced by `naive_randomisation_p` below - a second, deliberately
slow transcription of the definition that shares no code with the package. A test that
called into the optimised implementation to compute its own expectation would only show
that the implementation agrees with itself.

The definition being tested, for per-query differences d_i = a_i - b_i:

    observed  = mean(d)
    under H0, every sign assignment s in {-1, +1}^n is equally likely
    p         = |{s : |mean(s * d)| >= |observed|}| / 2**n     (two-sided)
"""

import itertools

import pytest

from groundwork.eval.significance import (
    holm_bonferroni,
    paired_randomization_test,
    paired_test_from_per_query,
)


def naive_randomisation_p(a, b):
    """Reference: enumerate all 2**n sign assignments, straight from the definition."""
    differences = [x - y for x, y in zip(a, b, strict=True)]
    n = len(differences)
    observed = sum(differences) / n

    at_least_as_extreme = 0
    for signs in itertools.product((-1, 1), repeat=n):
        mean = sum(s * d for s, d in zip(signs, differences, strict=True)) / n
        if abs(mean) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
    return at_least_as_extreme / (2**n)


class TestAgainstAnIndependentImplementation:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ([1.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ([0.5, 0.25, 0.75, 1.0], [0.25, 0.5, 0.5, 0.5]),
            ([0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.9, 0.7, 0.2, 0.6]),
            ([0.0] * 6, [1.0, 0.0, 0.5, 0.25, 0.75, 0.125]),
            ([0.31, 0.72, 0.05, 0.99, 0.44, 0.61, 0.18], [0.29, 0.70, 0.40, 0.11, 0.52, 0.60, 0.9]),
        ],
    )
    def test_exact_path_matches_naive_enumeration(self, a, b):
        result = paired_randomization_test(a, b)
        assert result.method == "exact"
        assert result.p_value == pytest.approx(naive_randomisation_p(a, b), abs=1e-12)

    def test_sampling_approximates_the_exact_answer(self):
        # 12 queries: small enough to enumerate exactly, large enough that sampling is
        # a real approximation. Forcing the sampled path by lowering the threshold lets
        # the two be compared on identical data.
        a = [0.10, 0.62, 0.35, 0.88, 0.41, 0.07, 0.93, 0.55, 0.22, 0.77, 0.48, 0.66]
        b = [0.05, 0.70, 0.30, 0.60, 0.55, 0.11, 0.80, 0.50, 0.40, 0.65, 0.52, 0.58]

        exact = paired_randomization_test(a, b, exact_max_queries=12)
        sampled = paired_randomization_test(
            a, b, exact_max_queries=4, num_resamples=200_000, seed=11
        )

        assert exact.method == "exact"
        assert sampled.method == "monte-carlo"
        # Monte Carlo standard error at p~0.3 over 200k draws is about 0.001, so 0.01
        # is a loose bound that still fails if the sampler is wrong.
        assert sampled.p_value == pytest.approx(exact.p_value, abs=0.01)


class TestHandComputedValues:
    def test_equal_positive_differences_only_the_unanimous_signs_are_extreme(self):
        # d = [0.2, 0.2, 0.2]; observed mean 0.2. |mean(s*d)| >= 0.2 requires all three
        # signs to agree, which happens for +++ and --- only: 2 of 2**3 = 8.
        result = paired_randomization_test([0.2, 0.2, 0.2], [0.0, 0.0, 0.0])
        assert result.p_value == pytest.approx(2 / 8)
        assert result.delta == pytest.approx(0.2)

    def test_unequal_differences_enumerated_by_hand(self):
        # d = [3, 1, 1]; observed sum 5. Signed sums: 5, 3, 3, 1, -1, -3, -3, -5.
        # Two of the eight have |sum| >= 5, so p = 2/8.
        result = paired_randomization_test([3.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        assert result.p_value == pytest.approx(2 / 8)

    def test_eight_equal_differences(self):
        # Same argument as above with n = 8: only ++++++++ and -------- qualify,
        # so p = 2 / 256 = 0.0078125. This is the floor an exact test can reach at
        # n = 8, which is why small query sets cannot produce small p-values.
        result = paired_randomization_test([1.0] * 8, [0.0] * 8)
        assert result.p_value == pytest.approx(2 / 256)

    def test_ties_contribute_nothing_and_are_counted(self):
        # d = [2, 1, 1, 0]; observed sum 4. The zero contributes nothing under either
        # sign, so the signed sums are those of [2, 1, 1] each occurring twice:
        # 4, 2, 2, 0, 0, -2, -2, -4 doubled. |sum| >= 4 for 4 of the 16 assignments.
        result = paired_randomization_test([2.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        assert result.p_value == pytest.approx(4 / 16)
        assert result.num_ties == 1
        assert result.effective_queries == 3

    def test_single_query_can_never_be_significant(self):
        # n = 1: both assignments give |mean| equal to |observed|, so p = 2/2 = 1.0.
        result = paired_randomization_test([1.0], [0.0])
        assert result.p_value == pytest.approx(1.0)

    def test_zero_mean_difference_scores_p_one(self):
        # d = [2, -1, -1]; observed mean 0, and every assignment has |mean| >= 0.
        result = paired_randomization_test([2.0, -1.0, -1.0], [0.0, 0.0, 0.0])
        assert result.delta == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)

    def test_identical_systems_are_indistinguishable(self):
        scores = [0.3, 0.9, 0.1, 0.4]
        result = paired_randomization_test(scores, scores)
        assert result.delta == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.num_ties == 4
        assert result.effective_queries == 0


class TestProperties:
    def test_two_sided_test_is_symmetric_in_its_arguments(self):
        a = [0.8, 0.2, 0.6, 0.4, 0.9]
        b = [0.3, 0.5, 0.6, 0.1, 0.2]
        forward = paired_randomization_test(a, b)
        backward = paired_randomization_test(b, a)
        assert forward.p_value == pytest.approx(backward.p_value)
        assert forward.delta == pytest.approx(-backward.delta)

    def test_sampled_path_is_reproducible_from_its_seed(self):
        a = [i / 40 for i in range(40)]
        b = [(i % 7) / 10 for i in range(40)]
        first = paired_randomization_test(a, b, num_resamples=5_000, seed=7)
        second = paired_randomization_test(a, b, num_resamples=5_000, seed=7)
        assert first.p_value == second.p_value
        assert first.seed == 7

    def test_exact_result_reports_no_seed_because_it_is_deterministic(self):
        result = paired_randomization_test([1.0, 2.0], [0.0, 0.0])
        assert result.seed is None
        assert result.num_assignments == 4

    def test_sampled_p_value_is_never_zero(self):
        # 40 queries where one system wins every single time: as separated as data can
        # be, yet no finite sample justifies claiming p = 0.
        a = [1.0] * 40
        b = [0.0] * 40
        result = paired_randomization_test(a, b, num_resamples=1_000, seed=3)
        assert result.method == "monte-carlo"
        assert result.p_value > 0.0
        assert result.p_value == pytest.approx(1 / 1001)

    def test_a_consistent_gap_is_less_likely_than_an_erratic_one(self):
        baseline = [0.5] * 12
        consistent = [0.6] * 12
        erratic = [0.6, 0.4, 0.7, 0.3, 0.6, 0.4, 0.7, 0.3, 0.6, 0.4, 0.7, 0.3]
        assert (
            paired_randomization_test(consistent, baseline).p_value
            < paired_randomization_test(erratic, baseline).p_value
        )


class TestInputValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            paired_randomization_test([1.0, 2.0], [1.0])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            paired_randomization_test([], [])

    def test_non_positive_resamples_raise(self):
        with pytest.raises(ValueError, match="num_resamples must be positive"):
            paired_randomization_test([1.0] * 20, [0.0] * 20, num_resamples=0)


class TestPairingByQueryId:
    def test_scores_are_aligned_by_query_id_not_insertion_order(self):
        a = {"q1": 1.0, "q2": 0.0, "q3": 0.5}
        shuffled_b = {"q3": 0.5, "q1": 0.0, "q2": 0.0}
        result = paired_test_from_per_query(a, shuffled_b)
        # Aligned by id the differences are q1: +1, q2: 0, q3: 0 -> delta 1/3.
        assert result.delta == pytest.approx(1 / 3)
        assert result.num_ties == 2

    def test_different_query_sets_raise_rather_than_silently_intersecting(self):
        with pytest.raises(ValueError, match="same queries"):
            paired_test_from_per_query({"q1": 1.0, "q2": 0.0}, {"q1": 1.0, "q3": 0.0})


class TestHolmBonferroni:
    def test_worked_example(self):
        # m = 3, sorted p: 0.01, 0.03, 0.04.
        #   rank 0: 3 * 0.01 = 0.03
        #   rank 1: 2 * 0.03 = 0.06
        #   rank 2: 1 * 0.04 = 0.04, raised to 0.06 to stay non-decreasing
        assert holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_adjusted_values_are_clipped_at_one(self):
        # m = 2, both 0.5: 2 * 0.5 = 1.0, then 1 * 0.5 = 0.5 raised to 1.0.
        assert holm_bonferroni([0.5, 0.5]) == pytest.approx([1.0, 1.0])

    def test_single_hypothesis_is_unchanged(self):
        assert holm_bonferroni([0.02]) == pytest.approx([0.02])

    def test_adjustment_never_lowers_a_p_value(self):
        raw = [0.001, 0.2, 0.04, 0.6]
        assert all(
            adjusted >= original
            for adjusted, original in zip(holm_bonferroni(raw), raw, strict=True)
        )

    def test_rejects_values_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm_bonferroni([0.5, 1.5])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="must not be empty"):
            holm_bonferroni([])
