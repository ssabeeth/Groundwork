"""Dataset loading."""

from groundwork.data.beir import (
    REFERENCE_NDCG_10,
    REFERENCE_TOLERANCE,
    BeirDataset,
    load_beir_dataset,
)

__all__ = [
    "REFERENCE_NDCG_10",
    "REFERENCE_TOLERANCE",
    "BeirDataset",
    "load_beir_dataset",
]
