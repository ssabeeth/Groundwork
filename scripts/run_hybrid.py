"""Fuse BM25 and dense retrieval with reciprocal rank fusion.

Usage::

    python scripts/run_hybrid.py --dataset nfcorpus --split train --sweep
    python scripts/run_hybrid.py --dataset nfcorpus --split test --k 60

The fusion constant ``k`` is a parameter like ``k1`` or ``alpha``, so it is swept on the
train split and the chosen value scored once on test. Tuning a fusion parameter on the
evaluation split is the most common way a hybrid result is overstated, and this repo has
already established the discipline that avoids it.

Dense embeddings are keyed by corpus rather than by split, and train and test share a
corpus, so the sweep reuses the cached matrix instead of re-encoding.
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
from groundwork.eval import evaluate_run, evaluate_run_per_query
from groundwork.retrieval import (
    LUCENE_ENGLISH_STOPWORDS,
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    RM3Retriever,
    Tokenizer,
    reciprocal_rank_fusion,
)
from groundwork.retrieval.dense import DEFAULT_MODEL
from groundwork.retrieval.rerank import DEFAULT_CROSS_ENCODER

SWEEP_K = (0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 200.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--systems",
        default="bm25,dense",
        help="Comma-separated systems to fuse: bm25, dense, rm3",
    )
    parser.add_argument("--k", type=float, default=60.0, help="RRF rank decay constant")
    parser.add_argument("--sweep", action="store_true", help="Sweep k on this split")
    parser.add_argument("--select-on", default="ndcg@10")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--fb-docs", type=int, default=5)
    parser.add_argument("--fb-terms", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--gain", default="exponential", choices=["exponential", "linear"])
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rescore the fused candidates with a cross-encoder",
    )
    parser.add_argument("--rerank-model", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_runs(args: argparse.Namespace, dataset) -> tuple[dict, dict]:  # noqa: ANN001
    """Retrieve with each requested system. Returns (runs_by_name, descriptions)."""
    names = [name.strip() for name in args.systems.split(",") if name.strip()]
    tokenizer = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)

    runs: dict[str, dict] = {}
    described: dict[str, dict] = {}

    if "bm25" in names or "rm3" in names:
        bm25 = BM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
        bm25.index(dataset.corpus)
        if "bm25" in names:
            runs["bm25"] = bm25.retrieve(dataset.queries, top_k=args.top_k)
            described["bm25"] = bm25.describe()
        if "rm3" in names:
            rm3 = RM3Retriever(
                fb_docs=args.fb_docs,
                fb_terms=args.fb_terms,
                alpha=args.alpha,
                k1=args.k1,
                b=args.b,
                tokenizer=tokenizer,
            )
            rm3.bm25 = bm25
            rm3._corpus = dataset.corpus
            runs["rm3"] = rm3.retrieve(dataset.queries, top_k=args.top_k)
            described["rm3"] = rm3.describe()

    if "dense" in names:
        dense = DenseRetriever(model_name=args.model)
        dense.index(dataset.corpus)
        runs["dense"] = dense.retrieve(dataset.queries, top_k=args.top_k)
        described["dense"] = dense.describe()

    missing = set(names) - set(runs)
    if missing:
        raise ValueError(f"unknown systems: {sorted(missing)}")
    return {name: runs[name] for name in names}, described


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading {args.dataset} ({args.split})")
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")

    start = time.perf_counter()
    runs, described = build_runs(args, dataset)
    build_seconds = time.perf_counter() - start

    for name, run in runs.items():
        parent = evaluate_run(run, dataset.qrels, k_values=(10, 100), gain=args.gain)
        nd, rc = parent["ndcg@10"], parent["recall@100"]
        print(f"  {name:<8} ndcg@10 {nd:.4f}  recall@100 {rc:.4f}")

    if args.sweep:
        cells = []
        for k in SWEEP_K:
            fused = reciprocal_rank_fusion(list(runs.values()), k=k, top_k=args.top_k)
            metrics = evaluate_run(fused, dataset.qrels, k_values=(10, 100), gain=args.gain)
            cells.append(
                {"k": k, "metrics": {a: b for a, b in metrics.items() if a != "num_queries"}}
            )
            nd, rc = metrics["ndcg@10"], metrics["recall@100"]
            print(f"  k={k:<6} ndcg@10 {nd:.4f}  recall@100 {rc:.4f}")
        best = max(cells, key=lambda cell: cell["metrics"][args.select_on])
        print(f"\n  best k on {args.select_on}: {best['k']}")
        record = {
            "dataset": args.dataset,
            "split": args.split,
            "method": "rrf",
            "systems": list(runs),
            "gain": args.gain,
            "select_on": args.select_on,
            "grid": list(SWEEP_K),
            "best": best,
            "cells": cells,
            "retrievers": described,
            "groundwork_version": __version__,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        output = (
            Path(args.output)
            if args.output
            else Path("results") / f"{args.dataset}-rrf-sweep-{args.split}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")
        return 0

    fused = reciprocal_rank_fusion(list(runs.values()), k=args.k, top_k=args.top_k)

    reranker_described = None
    if args.rerank:
        # Reranking reorders the candidate set; it cannot add to it. The fused run's
        # recall@100 is therefore a hard ceiling on what this can recover, which is why
        # it is applied after fusion rather than to BM25 alone.
        before = evaluate_run(fused, dataset.qrels, k_values=(10, 100), gain=args.gain)
        reranker = CrossEncoderReranker(model_name=args.rerank_model, depth=args.rerank_depth)
        start = time.perf_counter()
        fused = reranker.rerank(fused, dataset.queries, dataset.corpus)
        rerank_seconds = time.perf_counter() - start
        reranker_described = reranker.describe()
        after = evaluate_run(fused, dataset.qrels, k_values=(10, 100), gain=args.gain)
        print(
            f"  reranked {len(dataset.queries)} x {args.rerank_depth} pairs "
            f"in {rerank_seconds:.0f}s"
        )
        print(f"  ndcg@10   {before['ndcg@10']:.4f} -> {after['ndcg@10']:.4f}")
        print(f"  recall@100 {before['recall@100']:.4f} -> {after['recall@100']:.4f} (ceiling)")

    metrics_by_gain = {
        name: evaluate_run(fused, dataset.qrels, k_values=(1, 10, 100), gain=name)
        for name in ("exponential", "linear")
    }
    metrics = metrics_by_gain[args.gain]
    per_query = evaluate_run_per_query(fused, dataset.qrels, k_values=(1, 10, 100), gain=args.gain)
    levels_present = {level for d in dataset.qrels.values() for level in d.values() if level > 0}

    print(f"\n{args.dataset} / {args.split} / RRF({'+'.join(runs)}, k={args.k})")
    print("-" * 62)
    for name in sorted(metrics):
        if name != "num_queries":
            print(f"  {name:<14} {metrics[name]:.4f}")

    suffix = f"-{args.tag}" if args.tag else ""
    stem = "rrf-rerank" if args.rerank else "rrf"
    default_output = Path("results") / f"{args.dataset}-{stem}{suffix}.json"
    output = Path(args.output) if args.output else default_output
    per_query_path = output.parent / "per-query" / output.name

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": "rrf",
        "systems": list(runs),
        "rrf_k": args.k,
        "reranker": reranker_described,
        "metrics": {a: b for a, b in metrics.items() if a != "num_queries"},
        "num_queries": int(metrics["num_queries"]),
        "num_documents": len(dataset.corpus),
        "retrievers": described,
        "retriever": {"method": "rrf", "systems": list(runs), "k": args.k, "tokenizer": {}},
        "top_k": args.top_k,
        "gain": args.gain,
        "graded_qrels": len(levels_present) > 1,
        "relevance_levels": sorted(levels_present),
        "metrics_by_gain": {
            name: {a: b for a, b in scores.items() if a != "num_queries"}
            for name, scores in metrics_by_gain.items()
        },
        "timing_seconds": {"build": round(build_seconds, 2)},
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
                "retriever": record["retriever"],
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
