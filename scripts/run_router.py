"""Route each query to BM25 or to a dense encoder, and see whether it was worth it.

Usage::

    python scripts/run_router.py --dataset scifact \
        --lexical results/scifact-bm25.json \
        --semantic results/scifact-dense-bge.json \
        --lexical-train results/scifact-bm25-train.json \
        --semantic-train results/scifact-dense-bge-train.json

Experiment 8 measured a correlation: BM25's per-query advantage rises with the rarity of
the query's rarest term. This asks the question that correlation is only interesting for
— can you act on it? A router that picks the system per query using `max_idf` alone, with
its threshold chosen on train and scored once on test, either beats both systems or it
does not.

Nothing is retrieved here. Routing takes one system's whole ranking for a query, so the
routed run's score on that query *is* that system's score on it, and every per-query
score this needs is already committed under ``results/per-query/``. The index is rebuilt
only to compute ``max_idf``, which is a property of the corpus and not of any run.

Two baselines and a ceiling, and the ceiling is the important one:

- the better of the two systems being routed between, which a router must beat to exist;
- reciprocal rank fusion, which uses both systems on every query instead of choosing;
- the **oracle router**, which picks the better system per query by reading the labels.
  No router on any predictor can beat it. If the oracle sits barely above the better
  single system, routing is a dead end for every predictor at once, and that is a more
  useful thing to learn than whether this particular predictor works.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from groundwork import __version__
from groundwork.data import load_beir_dataset
from groundwork.retrieval import (
    LUCENE_ENGLISH_STOPWORDS,
    BM25Retriever,
    Tokenizer,
)
from groundwork.retrieval.routing import oracle_assignment, threshold_assignment

# Thresholds are quantiles of the predictor rather than absolute IDF values, because an
# absolute cut means something different on every corpus. 0.0 and 1.0 are included so the
# sweep always contains both single-system baselines and a router cannot win by being
# compared against a grid that excluded the things it has to beat.
QUANTILES = tuple(round(0.05 * step, 2) for step in range(21))

LEXICAL = "lexical"
SEMANTIC = "semantic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lexical", required=True, help="Lexical results file, test split")
    parser.add_argument("--semantic", required=True, help="Semantic results file, test split")
    parser.add_argument("--lexical-train", default=None, help="Lexical results file, train split")
    parser.add_argument("--semantic-train", default=None, help="Semantic results file, train split")
    parser.add_argument("--predictor", default="max_idf")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_per_query(path: Path, metric: str) -> dict[str, float]:
    """Per-query scores for ``metric`` from a results file."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    per_query = json.loads((path.parent / summary["per_query_file"]).read_text(encoding="utf-8"))[
        "per_query"
    ]
    return {qid: float(scores[metric]) for qid, scores in per_query.items()}


def routed_mean(scores: dict[str, dict[str, float]], assignment: dict[str, str]) -> float:
    """Mean score of the run that ``assignment`` describes."""
    return float(np.mean([scores[system][qid] for qid, system in assignment.items()]))


def main() -> int:
    args = parse_args()
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)

    test = {
        LEXICAL: load_per_query(Path(args.lexical), args.metric),
        SEMANTIC: load_per_query(Path(args.semantic), args.metric),
    }
    if test[LEXICAL].keys() != test[SEMANTIC].keys():
        raise ValueError("the two systems were scored on different query sets")

    # One index over the shared corpus. The predictor is a property of term rarity in the
    # corpus, so it does not depend on the split being scored, and is computed from a
    # concatenated index for the reason given in analyse_queries.py: it needs one IDF per
    # term, not one per field.
    tokenizer = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)
    index = BM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
    index.index(dataset.corpus)

    test_stats = {
        qid: index.query_statistics(dataset.queries[qid]) for qid in sorted(test[LEXICAL])
    }

    single = {name: float(np.mean(list(scores.values()))) for name, scores in test.items()}
    best_single = max(single, key=lambda name: single[name])
    oracle = oracle_assignment(test, args.metric)
    oracle_score = routed_mean(test, oracle)

    print(f"{args.dataset} / {args.split} / {args.metric}")
    print(f"  lexical   {single[LEXICAL]:.4f}  ({Path(args.lexical).name})")
    print(f"  semantic  {single[SEMANTIC]:.4f}  ({Path(args.semantic).name})")
    print(f"  oracle    {oracle_score:.4f}  (ceiling: reads the labels, not a method)")
    headroom = oracle_score - single[best_single]
    print(f"  headroom  {headroom:+.4f} over the better single system ({best_single})")

    record: dict[str, object] = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": "router",
        "predictor": args.predictor,
        "metric": args.metric,
        "lexical_run": args.lexical,
        "semantic_run": args.semantic,
        "num_queries": len(test[LEXICAL]),
        "single_system": single,
        "oracle": {
            "score": oracle_score,
            "headroom_over_best_single": headroom,
            "queries_routed_lexical": sum(1 for s in oracle.values() if s == LEXICAL),
        },
        "quantile_grid": list(QUANTILES),
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    if args.lexical_train and args.semantic_train:
        train_data = load_beir_dataset(args.dataset, split=args.train_split, data_dir=args.data_dir)
        train = {
            LEXICAL: load_per_query(Path(args.lexical_train), args.metric),
            SEMANTIC: load_per_query(Path(args.semantic_train), args.metric),
        }
        train_stats = {
            qid: index.query_statistics(train_data.queries[qid]) for qid in sorted(train[LEXICAL])
        }
        train_values = [values[args.predictor] for values in train_stats.values()]

        print(f"\n  sweeping {args.predictor} on {args.train_split} ({len(train_values)} queries)")
        cells = []
        for quantile in QUANTILES:
            threshold = float(np.quantile(train_values, quantile))
            assignment = threshold_assignment(
                train_stats, args.predictor, threshold, LEXICAL, SEMANTIC
            )
            cells.append(
                {
                    "quantile": quantile,
                    "threshold": threshold,
                    "score": routed_mean(train, assignment),
                    "queries_routed_lexical": sum(1 for s in assignment.values() if s == LEXICAL),
                }
            )
        best = max(cells, key=lambda cell: cell["score"])
        print(
            f"  best on train: quantile {best['quantile']}, "
            f"{args.predictor} >= {best['threshold']:.4f} -> {best['score']:.4f}"
        )

        chosen = threshold_assignment(
            test_stats, args.predictor, best["threshold"], LEXICAL, SEMANTIC
        )
        routed = routed_mean(test, chosen)
        print(
            f"  routed on {args.split}: {routed:.4f}  ({routed - single[best_single]:+.4f} "
            f"vs {best_single}, {routed - oracle_score:+.4f} vs oracle)"
        )

        record["train_sweep"] = {
            "split": args.train_split,
            "cells": cells,
            "best": best,
            "lexical_run": args.lexical_train,
            "semantic_run": args.semantic_train,
        }
        record["routed"] = {
            "score": routed,
            "threshold": best["threshold"],
            "delta_vs_best_single": routed - single[best_single],
            "delta_vs_oracle": routed - oracle_score,
            "queries_routed_lexical": sum(1 for s in chosen.values() if s == LEXICAL),
        }
    else:
        print("\n  no train split supplied: reporting the oracle ceiling only.")
        print("  A threshold chosen on the split it is scored on is not a result.")

    suffix = f"-{args.tag}" if args.tag else ""
    output = (
        Path(args.output)
        if args.output
        else Path("results") / f"{args.dataset}-router{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
