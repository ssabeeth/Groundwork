"""Evaluate RM3 pseudo-relevance feedback, tuning its parameters on a held-out split.

Usage::

    python scripts/run_rm3.py --dataset nfcorpus --sweep --split train
    python scripts/run_rm3.py --dataset nfcorpus --fb-docs 10 --fb-terms 20 --alpha 0.4

RM3 has three parameters and every one of them moves the result, so the same discipline
as experiment 3 applies: sweep on train, then score the chosen setting once on test. The
BM25 index is built once and shared across every cell, since feedback parameters change
only what is scored, never the index.
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
from tqdm import tqdm

from groundwork import __version__
from groundwork.data import load_beir_dataset
from groundwork.eval import evaluate_run, evaluate_run_per_query, oracle_recall_at_k
from groundwork.retrieval import LUCENE_ENGLISH_STOPWORDS, RM3Retriever, Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--fb-docs", type=int, default=10)
    parser.add_argument("--fb-terms", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--gain", default="exponential", choices=["exponential", "linear"])
    parser.add_argument("--sweep", action="store_true", help="Grid over the three parameters")
    parser.add_argument("--select-on", default="ndcg@10", help="Metric the sweep maximises")
    parser.add_argument("--tag", default="", help="Short label for this run")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


SWEEP_FB_DOCS = (3, 5, 10, 20)
SWEEP_FB_TERMS = (5, 10, 20, 50)
SWEEP_ALPHA = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)


def main() -> int:
    args = parse_args()

    print(f"Loading {args.dataset} ({args.split})")
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")

    tokenizer = Tokenizer(
        stopwords=None if args.no_stopwords else LUCENE_ENGLISH_STOPWORDS,
        stem=not args.no_stem,
    )
    base = RM3Retriever(
        fb_docs=args.fb_docs,
        fb_terms=args.fb_terms,
        alpha=args.alpha,
        k1=args.k1,
        b=args.b,
        tokenizer=tokenizer,
    )

    start = time.perf_counter()
    base.index(dataset.corpus)
    index_seconds = time.perf_counter() - start
    print(f"  indexed in {index_seconds:.1f}s")

    if args.sweep:
        grid = [(d, t, a) for d in SWEEP_FB_DOCS for t in SWEEP_FB_TERMS for a in SWEEP_ALPHA]
        cells = []
        start = time.perf_counter()
        for fb_docs, fb_terms, alpha in tqdm(grid, desc="Sweeping", unit="cell"):
            retriever = base.with_parameters(fb_docs=fb_docs, fb_terms=fb_terms, alpha=alpha)
            run = retriever.retrieve(dataset.queries, top_k=args.top_k, show_progress=False)
            metrics = evaluate_run(run, dataset.qrels, k_values=(10, 100), gain=args.gain)
            cells.append(
                {
                    "fb_docs": fb_docs,
                    "fb_terms": fb_terms,
                    "alpha": alpha,
                    "metrics": {k: v for k, v in metrics.items() if k != "num_queries"},
                }
            )
        sweep_seconds = time.perf_counter() - start

        best = max(cells, key=lambda cell: cell["metrics"][args.select_on])
        print(f"\n{args.dataset} / {args.split} / {len(grid)} cells in {sweep_seconds:.0f}s")
        print("-" * 62)
        print(
            f"  best on {args.select_on}: fb_docs={best['fb_docs']}, "
            f"fb_terms={best['fb_terms']}, alpha={best['alpha']}"
        )
        for name, value in sorted(best["metrics"].items()):
            print(f"    {name:<14} {value:.4f}")

        record = {
            "dataset": args.dataset,
            "split": args.split,
            "method": "rm3",
            "num_queries": len(dataset.queries),
            "gain": args.gain,
            "select_on": args.select_on,
            "grid": {
                "fb_docs": list(SWEEP_FB_DOCS),
                "fb_terms": list(SWEEP_FB_TERMS),
                "alpha": list(SWEEP_ALPHA),
            },
            "best": best,
            "cells": cells,
            "base_bm25": {"k1": args.k1, "b": args.b},
            "tokenizer": tokenizer.describe(),
            "timing_seconds": {"index": round(index_seconds, 2), "sweep": round(sweep_seconds, 2)},
            "groundwork_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        output = (
            Path(args.output)
            if args.output
            else Path("results") / f"{args.dataset}-rm3-sweep-{args.split}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")
        return 0

    start = time.perf_counter()
    run = base.retrieve(dataset.queries, top_k=args.top_k)
    retrieve_seconds = time.perf_counter() - start

    metrics_by_gain = {
        name: evaluate_run(run, dataset.qrels, k_values=(1, 10, 100), gain=name)
        for name in ("exponential", "linear")
    }
    metrics = metrics_by_gain[args.gain]
    per_query = evaluate_run_per_query(run, dataset.qrels, k_values=(1, 10, 100), gain=args.gain)
    levels_present = {level for d in dataset.qrels.values() for level in d.values() if level > 0}

    print(
        f"\n{args.dataset} / {args.split} / RM3 "
        f"(fb_docs={args.fb_docs}, fb_terms={args.fb_terms}, alpha={args.alpha}, gain={args.gain})"
    )
    print("-" * 62)
    for name in sorted(metrics):
        if name != "num_queries":
            print(f"  {name:<14} {metrics[name]:.4f}")
    print(f"  {'retrieve time':<14} {retrieve_seconds:.1f}s")

    suffix = f"-{args.tag}" if args.tag else ""
    default_output = Path("results") / f"{args.dataset}-rm3{suffix}.json"
    output = Path(args.output) if args.output else default_output
    per_query_path = output.parent / "per-query" / output.name

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": "rm3",
        "metrics": {k: v for k, v in metrics.items() if k != "num_queries"},
        "num_queries": int(metrics["num_queries"]),
        "num_documents": len(dataset.corpus),
        "oracle_recall_at_100": oracle_recall_at_k(dataset.qrels, 100),
        "retriever": base.describe(),
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
                "retriever": base.describe(),
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
