"""Paired significance testing for retrieval runs.

A difference in mean nDCG is not a result on its own. On 300 queries a gap of a few
tenths of a point is routinely produced by which queries happen to be in the benchmark
rather than by one system being better, and the only way to tell those apart is a test
that respects the pairing: both systems answered the *same* queries, so the per-query
differences are what carry the evidence.

The test implemented here is the **paired randomisation test**, also called a
permutation or sign-flip test. Smucker, Allan and Carterette (2007), "A Comparison of
Statistical Significance Tests for Information Retrieval Evaluation", compared the
usual candidates on exactly this problem and treat randomisation as the reference the
others are judged against. Three properties made it the choice here over a bootstrap or
a t-test:

1. It assumes nothing about the distribution of per-query scores. nDCG is bounded in
   [0, 1], frequently exactly 0 or exactly 1, and nothing like normal, which is
   precisely where a t-test's assumption is least comfortable.
2. It is *exactly* computable for small query counts by enumerating every sign
   assignment, so the sampled version can be checked against a ground truth rather
   than against itself. ``tests/test_significance.py`` does this.
3. It needs numpy and nothing else, so the core dependency list stays at two.

**The null hypothesis.** If the two systems are interchangeable, then for any query the
observed difference ``a_i - b_i`` was equally likely to have come out as ``b_i - a_i``.
So under the null, flipping the sign of any subset of the per-query differences is as
likely as the arrangement actually observed. Enumerating (or sampling) those sign
assignments builds the distribution of the mean difference under the null, and the
p-value is the share of it at least as extreme as what was measured.

**Two-sided by default**, because "is this different" is the honest question when
comparing two configurations. A one-sided test answers "is A better than B", which
presumes the direction before looking, and it is the wrong test for an ablation.

**What a p-value here does not tell you.** It speaks to whether a difference is
distinguishable from noise on this query set, not whether it is large enough to care
about, and not whether it generalises to another corpus. A significant 0.002 is still
0.002. Report the effect size alongside it, which is why ``delta`` is returned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Above this many queries, enumerating 2**n sign assignments stops being sensible
# (2**20 is already a million rows), so the test switches to sampling. The boundary is
# an argument rather than a constant because it changes which of two documented
# behaviours runs, and that belongs in the record of a result.
DEFAULT_EXACT_MAX_QUERIES = 16
DEFAULT_RESAMPLES = 10_000

Method = Literal["exact", "monte-carlo"]


@dataclass(frozen=True)
class SignificanceResult:
    """Outcome of a paired test.

    Attributes:
        delta: ``mean(a) - mean(b)``. Positive means the first system scored higher.
        p_value: Two-sided p-value. See the module docstring for its exact meaning.
        num_queries: Number of paired observations the test ran on.
        method: ``"exact"`` if every sign assignment was enumerated, ``"monte-carlo"``
            if they were sampled.
        num_assignments: Sign assignments considered: ``2**num_queries`` when exact,
            the resample count when sampled.
        seed: Random seed, or ``None`` when the result is exact and so deterministic.
        num_ties: Queries where the two systems scored identically. These contribute
            nothing under either sign and are reported because a comparison resting on
            a handful of non-tied queries deserves to be read with suspicion.
    """

    delta: float
    p_value: float
    num_queries: int
    method: Method
    num_assignments: int
    seed: int | None
    num_ties: int
    effective_queries: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_queries", self.num_queries - self.num_ties)

    def describe(self) -> dict[str, object]:
        """Settings and outcome, for recording alongside results."""
        return {
            "test": "paired_randomisation",
            "alternative": "two-sided",
            "delta": self.delta,
            "p_value": self.p_value,
            "num_queries": self.num_queries,
            "num_ties": self.num_ties,
            "effective_queries": self.effective_queries,
            "method": self.method,
            "num_assignments": self.num_assignments,
            "seed": self.seed,
        }


def _exact_sign_sums(differences: np.ndarray) -> np.ndarray:
    """Sum of every signed arrangement of ``differences``, all ``2**n`` of them."""
    n = differences.size
    assignments = np.arange(1 << n, dtype=np.int64)
    bits = (assignments[:, None] >> np.arange(n, dtype=np.int64)) & 1
    signs = 1.0 - 2.0 * bits
    return signs @ differences


def _sampled_sign_sums(differences: np.ndarray, num_resamples: int, seed: int) -> np.ndarray:
    """Sum of ``num_resamples`` randomly signed arrangements of ``differences``."""
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(num_resamples, differences.size))
    return signs @ differences


def paired_randomization_test(
    a: Sequence[float],
    b: Sequence[float],
    num_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    exact_max_queries: int = DEFAULT_EXACT_MAX_QUERIES,
) -> SignificanceResult:
    """Test whether two systems' per-query scores differ, respecting their pairing.

    ``a[i]`` and ``b[i]`` must be the two systems' scores on the *same* query, in the
    same order. Pairing them wrongly does not fail loudly — it quietly answers a
    different question — so callers should align by query id rather than by position;
    :func:`paired_test_from_per_query` does that.

    Args:
        a: Per-query scores for the first system.
        b: Per-query scores for the second system, aligned with ``a``.
        num_resamples: Sign assignments to sample when the query count is too large to
            enumerate. Ignored when the test runs exactly.
        seed: Seed for the sampled path, so a reported p-value can be reproduced.
        exact_max_queries: Enumerate every sign assignment at or below this many
            queries; sample above it.

    Returns:
        The test outcome, including the effect size it belongs with.

    Raises:
        ValueError: If the inputs differ in length, are empty, or the resample count
            is not positive.
    """
    if len(a) != len(b):
        raise ValueError(f"paired scores must be the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("paired scores must not be empty")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")

    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    differences = first - second

    n = differences.size
    observed_sum = float(differences.sum())
    delta = observed_sum / n
    num_ties = int(np.count_nonzero(differences == 0.0))

    # Comparisons are made on sums rather than means: same ordering, one fewer division,
    # and the tolerance below is easier to reason about. The tolerance exists because
    # the arrangement that reproduces the observed sum should always count as "at least
    # as extreme", and floating-point summation in a different order can land a few ulps
    # short of it.
    tolerance = 1e-12 * max(1.0, abs(observed_sum))

    if n <= exact_max_queries:
        sums = _exact_sign_sums(differences)
        method: Method = "exact"
        num_assignments = 1 << n
        used_seed: int | None = None
        at_least_as_extreme = int(np.count_nonzero(np.abs(sums) + tolerance >= abs(observed_sum)))
        p_value = at_least_as_extreme / num_assignments
    else:
        sums = _sampled_sign_sums(differences, num_resamples, seed)
        method = "monte-carlo"
        num_assignments = num_resamples
        used_seed = seed
        at_least_as_extreme = int(np.count_nonzero(np.abs(sums) + tolerance >= abs(observed_sum)))
        # The +1 on both sides counts the observed arrangement, which is always a valid
        # draw under the null. Without it a p-value of exactly 0 is reportable, and no
        # finite number of samples justifies that claim.
        p_value = (at_least_as_extreme + 1) / (num_resamples + 1)

    return SignificanceResult(
        delta=delta,
        p_value=min(1.0, p_value),
        num_queries=n,
        method=method,
        num_assignments=num_assignments,
        seed=used_seed,
        num_ties=num_ties,
    )


def paired_test_from_per_query(
    a: dict[str, float],
    b: dict[str, float],
    num_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    exact_max_queries: int = DEFAULT_EXACT_MAX_QUERIES,
) -> SignificanceResult:
    """Run :func:`paired_randomization_test` on two ``{query_id: score}`` mappings.

    Aligns the two systems by query id, which is the part that is easy to get wrong
    when scores are passed around as bare lists.

    Args:
        a: ``{query_id: score}`` for the first system.
        b: ``{query_id: score}`` for the second system.
        num_resamples: Passed through.
        seed: Passed through.
        exact_max_queries: Passed through.

    Returns:
        The test outcome.

    Raises:
        ValueError: If the two systems were not scored on the same query set. Testing
            the overlap instead would silently change the population being compared.
    """
    if a.keys() != b.keys():
        only_a = sorted(a.keys() - b.keys())
        only_b = sorted(b.keys() - a.keys())
        raise ValueError(
            "paired test needs the same queries on both sides; "
            f"{len(only_a)} only in the first (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in the second (e.g. {only_b[:3]})"
        )

    query_ids = sorted(a)
    return paired_randomization_test(
        [a[qid] for qid in query_ids],
        [b[qid] for qid in query_ids],
        num_resamples=num_resamples,
        seed=seed,
        exact_max_queries=exact_max_queries,
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Adjust p-values for testing several hypotheses on one dataset.

    A 2x2 ablation invites four or more comparisons off the same 300 queries, and at
    a 0.05 threshold roughly one in twenty independent tests clears it by chance alone.
    Holm-Bonferroni controls the probability of *any* false positive across the family
    while being uniformly less conservative than plain Bonferroni, and it assumes
    nothing about dependence between the tests, which matters here because comparisons
    sharing a configuration are certainly not independent.

    Args:
        p_values: Raw p-values, in any order.

    Returns:
        Adjusted p-values in the same order, each clipped to at most 1.0 and made
        non-decreasing in rank, which is what keeps the procedure coherent.

    Raises:
        ValueError: If the input is empty or contains a value outside [0, 1].
    """
    if not p_values:
        raise ValueError("p_values must not be empty")
    for p in p_values:
        if not 0.0 <= p <= 1.0 or math.isnan(p):
            raise ValueError(f"p-values must lie in [0, 1], got {p}")

    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (m - rank) * p_values[index]
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return adjusted


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values``, ties sharing their average rank.

    Average ranks are what makes Spearman well defined on data with ties, and per-query
    nDCG has a great many of them — a majority of queries often score exactly 0 or
    exactly 1. Assigning ties arbitrary distinct ranks would invent an ordering the data
    does not contain and bias the correlation.

    Args:
        values: One dimensional array.

    Returns:
        Ranks starting at 1.0, same shape as ``values``.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation between ``x`` and ``y``.

    Pearson correlation computed on average ranks, which is the definition that handles
    ties correctly. Rank-based because the quantities being related here — a query's
    term rarity and a difference in nDCG — have no reason to be linearly related, and
    only their monotone association is being claimed.

    Args:
        x: First variable.
        y: Second variable, paired with ``x`` by position.

    Returns:
        Correlation in [-1, 1], or 0.0 when either variable is constant and the
        correlation is undefined.

    Raises:
        ValueError: If the inputs differ in length or have fewer than two elements.
    """
    if len(x) != len(y):
        raise ValueError(f"paired inputs must be the same length, got {len(x)} and {len(y)}")
    if len(x) < 2:
        raise ValueError("correlation needs at least two observations")

    rank_x = _average_ranks(np.asarray(x, dtype=np.float64))
    rank_y = _average_ranks(np.asarray(y, dtype=np.float64))

    centred_x = rank_x - rank_x.mean()
    centred_y = rank_y - rank_y.mean()
    denominator = np.sqrt((centred_x**2).sum() * (centred_y**2).sum())
    if denominator == 0.0:
        return 0.0
    return float((centred_x * centred_y).sum() / denominator)


@dataclass(frozen=True)
class CorrelationResult:
    """Outcome of a permutation-tested correlation.

    Attributes:
        correlation: Spearman's rho.
        p_value: Two-sided p-value from the permutation test.
        num_observations: Number of paired observations.
        num_permutations: Permutations sampled.
        seed: Random seed used.
    """

    correlation: float
    p_value: float
    num_observations: int
    num_permutations: int
    seed: int

    def describe(self) -> dict[str, object]:
        """Settings and outcome, for recording alongside results."""
        return {
            "test": "spearman_permutation",
            "alternative": "two-sided",
            "correlation": self.correlation,
            "p_value": self.p_value,
            "num_observations": self.num_observations,
            "num_permutations": self.num_permutations,
            "seed": self.seed,
        }


def correlation_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    num_permutations: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> CorrelationResult:
    """Test whether two variables are associated, by permuting the pairing.

    Under the null hypothesis that ``x`` and ``y`` are unrelated, any pairing of one
    against the other is as likely as the pairing observed. Shuffling one of them
    repeatedly builds the distribution of the correlation under that null, with no
    appeal to a t-distribution whose assumptions per-query nDCG does not satisfy.

    Args:
        x: First variable.
        y: Second variable, paired with ``x`` by position.
        num_permutations: Shuffles to sample.
        seed: Seed, so a reported p-value can be reproduced.

    Returns:
        The correlation and its two-sided p-value.

    Raises:
        ValueError: If the inputs are mismatched, too short, or the permutation count
            is not positive.
    """
    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive")

    observed = spearman_correlation(x, y)

    # Ranks are permutation-invariant, so they are computed once and shuffled directly.
    rank_x = _average_ranks(np.asarray(x, dtype=np.float64))
    rank_y = _average_ranks(np.asarray(y, dtype=np.float64))
    centred_x = rank_x - rank_x.mean()
    centred_y = rank_y - rank_y.mean()
    scale = np.sqrt((centred_x**2).sum() * (centred_y**2).sum())

    if scale == 0.0:
        return CorrelationResult(0.0, 1.0, len(x), num_permutations, seed)

    rng = np.random.default_rng(seed)
    tolerance = 1e-12 * max(1.0, abs(observed))
    at_least_as_extreme = 0
    for _ in range(num_permutations):
        shuffled = rng.permutation(centred_y)
        candidate = float((centred_x * shuffled).sum() / scale)
        if abs(candidate) + tolerance >= abs(observed):
            at_least_as_extreme += 1

    # As in the paired test, the observed arrangement counts as one valid draw, so a
    # p-value of exactly zero is never reported.
    p_value = (at_least_as_extreme + 1) / (num_permutations + 1)
    return CorrelationResult(observed, min(1.0, p_value), len(x), num_permutations, seed)
