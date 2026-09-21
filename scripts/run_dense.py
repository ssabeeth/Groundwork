"""Score a dense bi-encoder on the same harness as everything else.

Usage::

    python scripts/run_dense.py --dataset nfcorpus

Writes a results file in the same shape as ``run_baseline.py``, so a dense run and a
BM25 run can be compared with ``compare_runs.py`` without either knowing what produced
the other. That is the whole point of the shared run shape.

Model id, maximum sequence length and truncation rate are recorded with every result.
The first is because a bi-encoder trained on data overlapping BEIR is not measuring what
a zero-shot benchmark claims to measure; the last is because "dense underperforms on
long documents" and "half the document was never encoded" are different findings.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from groundwork import __version__
from groundwork.data import load_beir_dataset
from groundwork.eval import evaluate_run, evaluate_run_per_query, oracle_recall_at_k
from groundwork.retrieval import DenseRetriever
from groundwork.retrieval.dense import DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=None,
        help="Split documents into word windows and score by the best",
    )
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--gain", default="exponential", choices=["exponential", "linear"])
    parser.add_argument("--cache-dir", default="data/embeddings")
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading {args.dataset} ({args.split})")
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")

    retriever = DenseRetriever(
        model_name=args.model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        chunk_words=args.chunk_words,
        chunk_overlap=args.chunk_overlap,
        cache_dir=args.cache_dir or None,
    )

    start = time.perf_counter()
    retriever.index(dataset.corpus)
    index_seconds = time.perf_counter() - start

    start = time.perf_counter()
    run = retriever.retrieve(dataset.queries, top_k=args.top_k)
    retrieve_seconds = time.perf_counter() - start

    metrics_by_gain = {
        name: evaluate_run(run, dataset.qrels, k_values=(1, 10, 100), gain=name)
        for name in ("exponential", "linear")
    }
    metrics = metrics_by_gain[args.gain]
    per_query = evaluate_run_per_query(run, dataset.qrels, k_values=(1, 10, 100), gain=args.gain)
    levels_present = {level for d in dataset.qrels.values() for level in d.values() if level > 0}

    print(f"\n{args.dataset} / {args.split} / dense ({args.model}, gain={args.gain})")
    print("-" * 62)
    for name in sorted(metrics):
        if name != "num_queries":
            print(f"  {name:<14} {metrics[name]:.4f}")
    truncation = retriever.truncation_rate
    print(f"  {'max_seq_len':<14} {retriever.effective_max_seq_length}")
    print(f"  {'truncated':<14} {'unknown' if truncation is None else f'{100 * truncation:.1f}%'}")
    print(f"  {'encode time':<14} {index_seconds:.1f}s")
    print(f"  {'retrieve time':<14} {retrieve_seconds:.1f}s")

    suffix = f"-{args.tag}" if args.tag else ""
    default_output = Path("results") / f"{args.dataset}-dense{suffix}.json"
    output = Path(args.output) if args.output else default_output
    per_query_path = output.parent / "per-query" / output.name

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": "dense",
        "metrics": {k: v for k, v in metrics.items() if k != "num_queries"},
        "num_queries": int(metrics["num_queries"]),
        "num_documents": len(dataset.corpus),
        "oracle_recall_at_100": oracle_recall_at_k(dataset.qrels, 100),
        "retriever": retriever.describe(),
        "top_k": args.top_k,
        "gain": args.gain,
        "graded_qrels": len(levels_present) > 1,
        "relevance_levels": sorted(levels_present),
        "metrics_by_gain": {
            name: {k: v for k, v in scores.items() if k != "num_queries"}
            for name, scores in metrics_by_gain.items()
        },
        "timing_seconds": {
            "encode": round(index_seconds, 2),
            "retrieve": round(retrieve_seconds, 2),
        },
        "per_query_file": str(per_query_path.relative_to(output.parent)),
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    per_query_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "tag": args.tag,
                "gain": args.gain,
                "retriever": retriever.describe(),
                "per_query": per_query,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output}")
    print(f"Wrote {per_query_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
