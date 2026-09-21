"""Score a learned sparse or late-interaction retriever on the same harness.

Usage::

    python scripts/run_learned.py --dataset scifact --method splade
    python scripts/run_learned.py --dataset nfcorpus --method colbert

Experiments 19 and 20. Both methods are neural, and neither is a bi-encoder: SPLADE
learns sparse term weights and scores with a dot product over an inverted index, ColBERT
keeps one vector per token and scores by MaxSim. One script covers both because the only
thing that differs is the retriever — the run shape, the metrics and the record are
identical to ``run_dense.py`` and ``run_baseline.py``, which is what lets any of them be
compared with ``compare_runs.py``.

The reference figure for the dataset is printed next to the measured one. Reproducing a
published number is the check that the implementation is right, and it has to come before
any comparison built on top of it, exactly as it did for BM25 in experiment 1.
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
from groundwork.retrieval import ColbertRetriever, SpladeRetriever
from groundwork.retrieval.late_interaction import DEFAULT_COLBERT_MODEL
from groundwork.retrieval.sparse import DEFAULT_SPLADE_MODEL

# Published nDCG@10 for these checkpoints, for the reproduction check only. Never used in
# a comparison: a published number was produced by a different implementation on different
# hardware, so it can say whether this harness is broken and nothing else.
PUBLISHED_NDCG_10 = {
    "splade": {"scifact": 0.693, "nfcorpus": 0.348, "trec-covid": 0.727},
    "colbert": {"scifact": 0.693, "nfcorpus": 0.338, "trec-covid": 0.738},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--method", choices=["splade", "colbert"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default=None, help="Defaults to the checkpoint for --method")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--gain", default="exponential", choices=["exponential", "linear"])
    parser.add_argument(
        "--max-length", type=int, default=None, help="SPLADE only; document token cap"
    )
    parser.add_argument(
        "--top-terms", type=int, default=None, help="SPLADE only; keep this many terms per document"
    )
    parser.add_argument(
        "--doc-length", type=int, default=None, help="ColBERT only; document token cap"
    )
    parser.add_argument(
        "--score-block",
        type=int,
        default=8192,
        help="ColBERT only; documents scored per block. Bounds memory, changes no score",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build(args: argparse.Namespace) -> SpladeRetriever | ColbertRetriever:
    """The retriever for ``--method``, with only that method's flags applied."""
    if args.method == "splade":
        return SpladeRetriever(
            model_name=args.model or DEFAULT_SPLADE_MODEL,
            batch_size=args.batch_size,
            top_terms=args.top_terms,
            **({"max_length": args.max_length} if args.max_length else {}),
        )
    return ColbertRetriever(
        model_name=args.model or DEFAULT_COLBERT_MODEL,
        batch_size=args.batch_size,
        score_block=args.score_block,
        **({"doc_length": args.doc_length} if args.doc_length else {}),
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading {args.dataset} ({args.split})")
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")

    retriever = build(args)

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

    described = retriever.describe()
    print(f"\n{args.dataset} / {args.split} / {args.method} ({described['model']})")
    print("-" * 62)
    for name in sorted(metrics):
        if name != "num_queries":
            print(f"  {name:<14} {metrics[name]:.4f}")

    published = PUBLISHED_NDCG_10.get(args.method, {}).get(args.dataset)
    delta = None
    if published is not None:
        delta = metrics["ndcg@10"] - published
        print(f"  {'published':<14} {published:.4f}  (delta {delta:+.4f})")
    print(f"  {'index time':<14} {index_seconds:.1f}s")
    print(f"  {'retrieve time':<14} {retrieve_seconds:.1f}s")

    suffix = f"-{args.tag}" if args.tag else ""
    default_output = Path("results") / f"{args.dataset}-{args.method}{suffix}.json"
    output = Path(args.output) if args.output else default_output
    per_query_path = output.parent / "per-query" / output.name

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": args.method,
        "metrics": {k: v for k, v in metrics.items() if k != "num_queries"},
        "num_queries": int(metrics["num_queries"]),
        "num_documents": len(dataset.corpus),
        "oracle_recall_at_100": oracle_recall_at_k(dataset.qrels, 100),
        "retriever": described,
        "published_ndcg_at_10": published,
        "delta_vs_published_ndcg_at_10": None if delta is None else round(delta, 4),
        "top_k": args.top_k,
        "gain": args.gain,
        "graded_qrels": len(levels_present) > 1,
        "relevance_levels": sorted(levels_present),
        "metrics_by_gain": {
            name: {k: v for k, v in scores.items() if k != "num_queries"}
            for name, scores in metrics_by_gain.items()
        },
        "timing_seconds": {
            "index": round(index_seconds, 2),
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
                "retriever": described,
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
