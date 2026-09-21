"""Evaluation metrics and harness."""

from groundwork.eval.metrics import (
    evaluate_run,
    evaluate_run_per_query,
    ndcg_at_k,
    recall_at_k,
)
from groundwork.eval.significance import (
    SignificanceResult,
    holm_bonferroni,
    paired_randomization_test,
    paired_test_from_per_query,
)

__all__ = [
    "SignificanceResult",
    "evaluate_run",
    "evaluate_run_per_query",
    "holm_bonferroni",
    "ndcg_at_k",
    "paired_randomization_test",
    "paired_test_from_per_query",
    "recall_at_k",
]
