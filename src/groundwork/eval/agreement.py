"""How much two assessors agree, corrected for agreement by chance.

Written for one purpose: checking an LLM judge against the human relevance assessors who
produced the qrels. "LLM-as-judge" is the standard way RAG systems are evaluated in 2026
and it is almost never calibrated, so a judge's verdict is usually reported as though it
were ground truth rather than as one more measurement with an error rate.

Raw agreement is not enough, and the reason matters here. Relevance judgements are heavily
skewed: in a BEIR pool most query-document pairs are non-relevant, so a judge that answers
"not relevant" to everything scores upwards of 90% raw agreement while carrying no
information at all. Cohen's kappa subtracts the agreement two assessors would reach by
chance given their marginal rates, which is exactly the failure mode a skewed pool
produces.

    kappa = (p_observed - p_chance) / (1 - p_chance)

Kappa is 1 for perfect agreement, 0 for chance-level, and negative when two assessors
disagree more than chance would predict. It is reported here alongside raw agreement and
the confusion counts, because kappa alone hides which direction a judge is wrong in, and
"calls irrelevant documents relevant" and "misses relevant documents" have very different
consequences for a retrieval evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgreementResult:
    """Agreement between two assessors over the same items."""

    num_items: int
    raw_agreement: float
    cohens_kappa: float
    chance_agreement: float
    labels: list[int]
    confusion: list[list[int]]

    def describe(self) -> dict[str, object]:
        """A JSON-safe record, for writing beside results."""
        return {
            "test": "cohens_kappa",
            "num_items": self.num_items,
            "raw_agreement": self.raw_agreement,
            "chance_agreement": self.chance_agreement,
            "cohens_kappa": self.cohens_kappa,
            "labels": self.labels,
            "confusion": self.confusion,
        }


def confusion_matrix(first: Sequence[int], second: Sequence[int]) -> tuple[list[int], np.ndarray]:
    """Counts of every (first, second) label pair.

    Args:
        first: One assessor's labels.
        second: The other assessor's labels, aligned item for item.

    Returns:
        ``(labels, matrix)`` where ``labels`` is the sorted union of labels seen and
        ``matrix[i][j]`` counts items the first assessor called ``labels[i]`` and the
        second called ``labels[j]``.

    Raises:
        ValueError: If the two sequences differ in length, which would silently align
            different items with each other.
    """
    if len(first) != len(second):
        raise ValueError(
            f"assessors labelled different numbers of items: {len(first)} vs {len(second)}"
        )

    labels = sorted(set(first) | set(second))
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(first, second, strict=True):
        matrix[index[a], index[b]] += 1
    return labels, matrix


def cohens_kappa(first: Sequence[int], second: Sequence[int]) -> AgreementResult:
    """Chance-corrected agreement between two assessors.

    Args:
        first: One assessor's labels.
        second: The other assessor's labels, aligned item for item.

    Returns:
        An :class:`AgreementResult` carrying kappa, raw agreement and the confusion
        counts.

    Raises:
        ValueError: If there are no items, or the sequences differ in length.

    Note:
        When both assessors give every item the same single label, chance agreement is
        1.0 and kappa is undefined — the formula divides by zero. That case returns
        kappa 0.0, which is the honest reading: two assessors who only ever say one
        thing have demonstrated no agreement beyond chance, however often they match.
    """
    if not len(first):
        raise ValueError("no items to compare")

    labels, matrix = confusion_matrix(first, second)
    total = matrix.sum()
    observed = float(np.trace(matrix)) / total

    row_totals = matrix.sum(axis=1) / total
    column_totals = matrix.sum(axis=0) / total
    chance = float(np.dot(row_totals, column_totals))

    kappa = 0.0 if chance >= 1.0 else (observed - chance) / (1.0 - chance)
    return AgreementResult(
        num_items=int(total),
        raw_agreement=observed,
        cohens_kappa=float(kappa),
        chance_agreement=chance,
        labels=[int(label) for label in labels],
        confusion=matrix.tolist(),
    )
