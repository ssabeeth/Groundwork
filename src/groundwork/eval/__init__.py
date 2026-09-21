"""Evaluation metrics and harness."""

from groundwork.eval.agreement import (
    AgreementResult,
    cohens_kappa,
    confusion_matrix,
)
from groundwork.eval.metrics import (
    evaluate_run,
    evaluate_run_per_query,
    ndcg_at_k,
    oracle_recall_at_k,
    recall_at_k,
)
from groundwork.eval.significance import (
    CorrelationResult,
    SignificanceResult,
    correlation_permutation_test,
    holm_bonferroni,
    paired_randomization_test,
    paired_test_from_per_query,
    spearman_correlation,
)

__all__ = [
    "AgreementResult",
    "CorrelationResult",
    "SignificanceResult",
    "cohens_kappa",
    "confusion_matrix",
    "correlation_permutation_test",
    "evaluate_run",
    "evaluate_run_per_query",
    "holm_bonferroni",
    "ndcg_at_k",
    "oracle_recall_at_k",
    "paired_randomization_test",
    "paired_test_from_per_query",
    "recall_at_k",
    "spearman_correlation",
]
