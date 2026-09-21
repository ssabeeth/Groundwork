"""Measure how much a dataset's choice of query formulation moves the score.

Usage::

    python scripts/diagnose_query_fields.py --dataset trec-covid

Most BEIR datasets ship one query per id and there is nothing to choose. TREC-COVID
ships three — a question in ``text``, a keyword form and a narrative in ``metadata`` —
and the choice between them moves nDCG@10 by more than tokenisation, `k1`/`b` and the
gain function combined. Experiment 4 found this while investigating why TREC-COVID does
not reproduce its published number, and this script exists so that finding is produced
by the repository rather than by a command someone once ran.

The index is built once and each formulation re-retrieved against it, since the index
does not depend on the queries.

This is a diagnostic, not a tuning loop. The loader deliberately keeps using the ``text``
field, which is what BEIR designates as the query for every dataset. Selecting whichever
formulation best matched a published figure would make the reproduction unfalsifiable.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from groundwork import __version__
from groundwork.data import REFERENCE_NDCG_10, load_beir_dataset
from groundwork.eval import evaluate_run
from groundwork.retrieval import (
    LUCENE_ENGLISH_STOPWORDS,
    BM25Retriever,
    MultiFieldBM25Retriever,
    Tokenizer,
)

# How each formulation is assembled from a queries.jsonl record.
FORMULATIONS = {
    "text": ("text",),
    "metadata.query": ("metadata.query",),
    "metadata.narrative": ("metadata.narrative",),
    "query+text": ("metadata.query", "text"),
    "query+text+narrative": ("metadata.query", "text", "metadata.narrative"),
}


def field(record: dict, path: str) -> str:
    """Fetch a dotted field from a queries.jsonl record, or "" when absent."""
    value: object = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return ""
        value = value[key]
    return str(value) if value is not None else ""


# Each formulation is reported against the published BM25 figure rather than against the
# canonical one, because the question here is how far a formulation choice moves you from
# the number BEIR reports. Read from run_baseline.py so there is one definition of it.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="trec-covid")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--gain", default="exponential", choices=["exponential", "linear"])
    parser.add_argument(
        "--single-field",
        action="store_true",
        help="Concatenate title and text (the pre-experiment-10 default)",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    raw: dict[str, dict] = {}
    queries_path = Path(args.data_dir) / args.dataset / "queries.jsonl"
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                raw[record["_id"]] = record

    available = {
        name: paths
        for name, paths in FORMULATIONS.items()
        if all(any(field(raw[qid], path) for qid in dataset.queries) for path in paths)
    }
    print(f"{args.dataset}: {len(available)} of {len(FORMULATIONS)} formulations present")

    tokenizer = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)
    retriever = (
        BM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
        if args.single_field
        else MultiFieldBM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
    )
    start = time.perf_counter()
    retriever.index(dataset.corpus)
    index_seconds = time.perf_counter() - start
    print(f"  indexed {len(dataset.corpus)} documents in {index_seconds:.0f}s\n")

    print(f"  {'formulation':<24} {'nDCG@10':>9} {'recall@100':>11}")
    variants = []
    for name, paths in available.items():
        queries = {
            qid: " ".join(part for part in (field(raw[qid], p) for p in paths) if part).strip()
            for qid in dataset.queries
        }
        run = retriever.retrieve(queries, top_k=args.top_k, show_progress=False)
        metrics = evaluate_run(run, dataset.qrels, k_values=(10, 100), gain=args.gain)
        variants.append(
            {
                "formulation": name,
                "fields": list(paths),
                "metrics": {k: v for k, v in metrics.items() if k != "num_queries"},
            }
        )
        print(f"  {name:<24} {metrics['ndcg@10']:>9.4f} {metrics['recall@100']:>11.4f}")

    # Each formulation's gap against the one BEIR actually uses. The point of this
    # diagnostic is the spread between formulations, so the gaps are recorded rather than
    # subtracted in prose later.
    published = REFERENCE_NDCG_10.get(args.dataset)
    if published is not None:
        for variant in variants:
            variant["delta_vs_published_ndcg_at_10"] = variant["metrics"]["ndcg@10"] - published

    scores = [v["metrics"]["ndcg@10"] for v in variants]
    spread = max(scores) - min(scores)
    print(f"\n  spread across formulations: {spread:.4f} nDCG@10")

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "num_queries": len(dataset.queries),
        "num_documents": len(dataset.corpus),
        "retriever": retriever.describe(),
        "gain": args.gain,
        "top_k": args.top_k,
        "variants": variants,
        "ndcg_at_10_spread": spread,
        "loader_uses": "text",
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    output = (
        Path(args.output) if args.output else Path("results") / f"{args.dataset}-query-fields.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
