"""Score BM25 over expanded queries or expanded documents.

Usage::

    python scripts/run_expanded.py --dataset nfcorpus --kind query --split train --sweep
    python scripts/run_expanded.py --dataset nfcorpus --kind query --weight 5 --tag hyde
    python scripts/run_expanded.py --dataset nfcorpus --kind document --tag doc2query

Reads generated text written by ``generate_expansions.py`` and applies it. No model is
loaded here, so a weight sweep costs seconds rather than an hour of generation.

Query expansion has one parameter — how many times the original query is repeated beside
the generated passage — and it is swept on train and scored once on test, like `k1`,
`alpha` and the fusion constant before it. Document expansion has none: the generations
are fixed and either help or do not.

The comparison that matters is not against BM25. RM3 is already measured on this harness
and closes the same vocabulary gap by reading the corpus instead of a model's parameters,
so it is the number a generative expander has to beat to have earned its dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from groundwork import __version__
from groundwork.data import load_beir_dataset
from groundwork.eval import evaluate_run, evaluate_run_per_query, oracle_recall_at_k
from groundwork.retrieval import (
    LUCENE_ENGLISH_STOPWORDS,
    BM25Retriever,
    MultiFieldBM25Retriever,
    Tokenizer,
)
from groundwork.retrieval.expansion import expand_documents, expand_queries

SWEEP_WEIGHTS = (0, 1, 2, 3, 5, 8, 12, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--kind", choices=["query", "document"], required=True)
    parser.add_argument("--expansions", default=None, help="Defaults to the generated file")
    parser.add_argument(
        "--weight",
        type=int,
        default=5,
        help="Times to repeat the original query beside the generated passage",
    )
    parser.add_argument("--sweep", action="store_true", help="Sweep the weight on this split")
    parser.add_argument("--select-on", default="ndcg@10")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--gain", default="exponential", choices=["exponential", "linear"])
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--single-field", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_expansions(path: Path) -> dict:
    """The generated text and the settings that produced it."""
    record = json.loads(path.read_text(encoding="utf-8"))
    return record


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)

    # Expansions are generated per split for queries and once per corpus for documents,
    # since a corpus does not have a split.
    default_split = args.split if args.kind == "query" else "test"
    expansion_path = (
        Path(args.expansions)
        if args.expansions
        else Path(args.data_dir) / "expansions" / f"{args.dataset}-{args.kind}-{default_split}.json"
    )
    if not expansion_path.exists():
        raise FileNotFoundError(
            f"{expansion_path} does not exist. Run generate_expansions.py --dataset "
            f"{args.dataset} --kind {args.kind} first."
        )
    expansion_record = load_expansions(expansion_path)
    expansions = expansion_record["expansions"]

    tokenizer = Tokenizer(
        stopwords=None if args.no_stopwords else LUCENE_ENGLISH_STOPWORDS,
        stem=not args.no_stem,
    )

    def build(corpus) -> object:  # noqa: ANN001 - corpus is a plain mapping
        retriever = (
            BM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
            if args.single_field
            else MultiFieldBM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
        )
        retriever.index(corpus)
        return retriever

    print(f"Loading {args.dataset} ({args.split})")
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")
    print(f"  expansions from {expansion_path.name} ({expansion_record['model']})")

    start = time.perf_counter()
    if args.kind == "document":
        corpus = expand_documents(dataset.corpus, expansions)
        queries = dict(dataset.queries)
        grew = sum(
            1
            for doc_id in dataset.corpus
            if corpus[doc_id]["text"] != dataset.corpus[doc_id].get("text", "")
        )
        print(f"  expanded {grew} of {len(corpus)} documents")
    else:
        corpus = dataset.corpus
        # Query expansions are stored as a list per id, like document ones, so that one
        # file format serves both. Query expansion uses the first generation.
        flat = {qid: values[0] if values else "" for qid, values in expansions.items()}
        queries = expand_queries(dataset.queries, flat, weight=args.weight)
    retriever = build(corpus)
    build_seconds = time.perf_counter() - start

    if args.sweep:
        if args.kind != "query":
            raise ValueError("--sweep applies to query expansion; document expansion has no weight")
        cells = []
        for weight in SWEEP_WEIGHTS:
            flat = {qid: values[0] if values else "" for qid, values in expansions.items()}
            swept = expand_queries(dataset.queries, flat, weight=weight)
            run = retriever.retrieve(swept, top_k=args.top_k, show_progress=False)
            metrics = evaluate_run(run, dataset.qrels, k_values=(10, 100), gain=args.gain)
            cells.append(
                {
                    "weight": weight,
                    "metrics": {a: b for a, b in metrics.items() if a != "num_queries"},
                }
            )
            nd, rc = metrics["ndcg@10"], metrics["recall@100"]
            print(f"  weight={weight:<4} ndcg@10 {nd:.4f}  recall@100 {rc:.4f}")
        best = max(cells, key=lambda cell: cell["metrics"][args.select_on])
        print(f"\n  best weight on {args.select_on}: {best['weight']}")

        # A one-dimensional sweep has two failure modes that reporting only the best cell
        # hides. The peak can sit on the edge of the grid, which means the grid was too
        # narrow to contain the optimum. And the curve can never turn over, which means
        # the sweep is asking for more of something the grid caps.
        #
        # Both matter here more than usual, because of what the parameter is. `weight` is
        # how many times the ORIGINAL query is repeated before the generated passage, so
        # a larger weight dilutes the expansion. A curve that rises monotonically to the
        # largest weight on the grid is the tuner asking for *less* expansion, and its
        # limit is the unexpanded baseline rather than any setting of this method.
        selected = [cell["metrics"][args.select_on] for cell in cells]
        peak = max(selected)
        # pairwise, not zip(xs, xs[1:], strict=True): the offset pairing is deliberate,
        # so strict= raises on the length mismatch - but only once all() consumes the
        # whole iterator, which happens precisely when the curve IS monotone.
        rising = all(b > a for a, b in pairwise(selected))
        falling = all(b < a for a, b in pairwise(selected))
        surface = {
            "metric": args.select_on,
            "num_cells": len(cells),
            "best": peak,
            "worst": min(selected),
            "span": peak - min(selected),
            "best_weight": best["weight"],
            "best_at_grid_edge": best["weight"] in (min(SWEEP_WEIGHTS), max(SWEEP_WEIGHTS)),
            "monotone": "increasing" if rising else "decreasing" if falling else "neither",
        }
        print(f"  span across {len(cells)} weights  {surface['span']:.4f}")
        if surface["best_at_grid_edge"]:
            print(
                f"  NOTE: the best weight is at the edge of the grid "
                f"{min(SWEEP_WEIGHTS)}..{max(SWEEP_WEIGHTS)}, and the curve is "
                f"{surface['monotone']}. The optimum may lie outside the grid; for this "
                f"parameter that direction means less expansion, not more."
            )
        record = {
            "dataset": args.dataset,
            "split": args.split,
            "method": "query-expansion",
            "kind": args.kind,
            "gain": args.gain,
            "select_on": args.select_on,
            "grid": list(SWEEP_WEIGHTS),
            "best": best,
            "surface": surface,
            "cells": cells,
            "expansion": {k: v for k, v in expansion_record.items() if k != "expansions"},
            "retriever": retriever.describe(),
            "groundwork_version": __version__,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        output = (
            Path(args.output)
            if args.output
            else Path("results") / f"{args.dataset}-{args.kind}-expansion-sweep-{args.split}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")
        return 0

    run = retriever.retrieve(queries, top_k=args.top_k)
    metrics_by_gain = {
        name: evaluate_run(run, dataset.qrels, k_values=(1, 10, 100), gain=name)
        for name in ("exponential", "linear")
    }
    metrics = metrics_by_gain[args.gain]
    per_query = evaluate_run_per_query(run, dataset.qrels, k_values=(1, 10, 100), gain=args.gain)
    levels = {level for d in dataset.qrels.values() for level in d.values() if level > 0}

    print(f"\n{args.dataset} / {args.split} / {args.kind} expansion")
    print("-" * 62)
    for name in sorted(metrics):
        if name != "num_queries":
            print(f"  {name:<14} {metrics[name]:.4f}")

    suffix = f"-{args.tag}" if args.tag else ""
    output = (
        Path(args.output)
        if args.output
        else Path("results") / f"{args.dataset}-{args.kind}-expansion{suffix}.json"
    )
    per_query_path = output.parent / "per-query" / output.name

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": f"{args.kind}-expansion",
        "query_weight": args.weight if args.kind == "query" else None,
        "metrics": {a: b for a, b in metrics.items() if a != "num_queries"},
        "num_queries": int(metrics["num_queries"]),
        "num_documents": len(corpus),
        "oracle_recall_at_100": oracle_recall_at_k(dataset.qrels, 100),
        "expansion": {k: v for k, v in expansion_record.items() if k != "expansions"},
        "retriever": retriever.describe(),
        "top_k": args.top_k,
        "gain": args.gain,
        "graded_qrels": len(levels) > 1,
        "relevance_levels": sorted(levels),
        "metrics_by_gain": {
            name: {a: b for a, b in scores.items() if a != "num_queries"}
            for name, scores in metrics_by_gain.items()
        },
        "gain_difference_ndcg_at_10": (
            metrics_by_gain["exponential"]["ndcg@10"] - metrics_by_gain["linear"]["ndcg@10"]
        ),
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
