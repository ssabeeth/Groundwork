"""Sweep BM25's k1 and b on one split, so a setting can be chosen without peeking.

Usage::

    python scripts/run_sweep.py --dataset scifact --split train

Answering "are BEIR's k1=0.9, b=0.4 right for this corpus" by sweeping on the test set
and reporting the best cell does not answer it: the winner of a large grid is partly
winning by chance, and its margin over the default is inflated by the same selection
that picked it. SciFact ships a train split over the identical corpus, so the sweep runs
there and the chosen setting is then scored once on test, which stays untouched.

The index is built once and shared across every cell. k1 and b appear only in scoring,
never in the postings, lengths or IDF, so rebuilding per cell would be pure waste.
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
from groundwork.eval import evaluate_run
from groundwork.retrieval import LUCENE_ENGLISH_STOPWORDS, BM25Retriever, Tokenizer

BEIR_DEFAULT_K1 = 0.9
BEIR_DEFAULT_B = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--split", default="train", help="Split to sweep on")
    parser.add_argument("--data-dir", default="data", help="Where datasets are cached")
    parser.add_argument("--k1-min", type=float, default=0.2)
    parser.add_argument("--k1-max", type=float, default=2.0)
    parser.add_argument("--k1-step", type=float, default=0.2)
    parser.add_argument("--b-min", type=float, default=0.0)
    parser.add_argument("--b-max", type=float, default=1.0)
    parser.add_argument("--b-step", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--no-stem", action="store_true", help="Disable Porter stemming")
    parser.add_argument("--no-stopwords", action="store_true", help="Keep stopwords")
    parser.add_argument(
        "--select-on",
        default="ndcg@10",
        help="Metric whose best cell is reported as the chosen setting",
    )
    parser.add_argument("--output", default=None, help="Results JSON path")
    return parser.parse_args()


def frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range, rounded to avoid 0.30000000000000004 in output."""
    if step <= 0:
        raise ValueError("step must be positive")
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(count)]


def main() -> int:
    args = parse_args()

    print(f"Loading {args.dataset} ({args.split})")
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")

    tokenizer = Tokenizer(
        stopwords=None if args.no_stopwords else LUCENE_ENGLISH_STOPWORDS,
        stem=not args.no_stem,
    )

    base = BM25Retriever(k1=BEIR_DEFAULT_K1, b=BEIR_DEFAULT_B, tokenizer=tokenizer)
    start = time.perf_counter()
    base.index(dataset.corpus)
    index_seconds = time.perf_counter() - start
    print(f"  indexed once in {index_seconds:.1f}s; every cell reuses it")

    # The default being judged has to be on the grid, or there is nothing to compare the
    # winner against. A step size that happens to miss 0.9 is easy to choose by accident.
    k1_values = sorted(set(frange(args.k1_min, args.k1_max, args.k1_step)) | {BEIR_DEFAULT_K1})
    b_values = sorted(set(frange(args.b_min, args.b_max, args.b_step)) | {BEIR_DEFAULT_B})
    grid = [(k1, b) for k1 in k1_values for b in b_values]

    cells = []
    start = time.perf_counter()
    for k1, b in tqdm(grid, desc="Sweeping", unit="cell"):
        retriever = base.with_parameters(k1=k1, b=b)
        run = retriever.retrieve(dataset.queries, top_k=args.top_k, show_progress=False)
        metrics = evaluate_run(run, dataset.qrels, k_values=(10, 100))
        scores = {name: value for name, value in metrics.items() if name != "num_queries"}
        cells.append({"k1": k1, "b": b, "metrics": scores})
    sweep_seconds = time.perf_counter() - start

    if args.select_on not in cells[0]["metrics"]:
        raise KeyError(f"--select-on {args.select_on!r} not among {sorted(cells[0]['metrics'])}")

    best = max(cells, key=lambda cell: cell["metrics"][args.select_on])
    default = next(
        (cell for cell in cells if cell["k1"] == BEIR_DEFAULT_K1 and cell["b"] == BEIR_DEFAULT_B),
        None,
    )
    if default is None:  # pragma: no cover - the grid is constructed to contain it
        raise RuntimeError(
            f"k1={BEIR_DEFAULT_K1}, b={BEIR_DEFAULT_B} missing from the grid; "
            "nothing to compare the best cell against"
        )

    print(f"\n{args.dataset} / {args.split} / {len(grid)} cells in {sweep_seconds:.1f}s")
    print("-" * 62)
    print(f"  best on {args.select_on:<12} k1={best['k1']}, b={best['b']}")
    for name, value in sorted(best["metrics"].items()):
        print(f"    {name:<14} {value:.4f}")
    print(f"  BEIR default          k1={default['k1']}, b={default['b']}")
    for name, value in sorted(default["metrics"].items()):
        print(f"    {name:<14} {value:.4f}")
    gap = best["metrics"][args.select_on] - default["metrics"][args.select_on]
    print(f"  gap on {args.select_on:<13} {gap:+.4f}  (on the tuning split, so optimistic)")

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "num_queries": len(dataset.queries),
        "num_documents": len(dataset.corpus),
        "tokenizer": tokenizer.describe(),
        "top_k": args.top_k,
        "select_on": args.select_on,
        "grid": {"k1": k1_values, "b": b_values},
        "best": best,
        "beir_default": default,
        "cells": cells,
        "timing_seconds": {"index": round(index_seconds, 2), "sweep": round(sweep_seconds, 2)},
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    output = (
        Path(args.output)
        if args.output
        else Path("results") / f"{args.dataset}-bm25-sweep-{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
