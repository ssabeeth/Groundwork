"""Test whether lexical retrieval wins on lexically specific queries.

Usage::

    python scripts/analyse_queries.py --dataset nfcorpus \
        --lexical results/nfcorpus-bm25-graded.json \
        --semantic results/nfcorpus-dense.json

This is the project's central claim put in a form that can be wrong.

**The hypothesis, fixed before looking at any result.** BM25 scores a document by the
query terms it literally contains, weighted by how rare those terms are. A query
containing a genuinely rare token — a gene symbol, an accession number, a species
binomial — therefore hands BM25 something close to a unique key, while a bi-encoder
maps it into a neighbourhood of things that keep similar company. So:

    per-query (nDCG_lexical - nDCG_semantic) should correlate POSITIVELY
    with the rarity of the query's rarest term.

**Why one continuous predictor rather than query-type buckets.** Labelling queries by
type and comparing group means invites picking the labelling that shows an effect, and
with enough candidate groupings one always will. A single predictor, specified in
advance, computable from the query and the index alone before any retrieval is run,
gives one hypothesis and one test. `max_idf` is that predictor; `mean_idf` and the
out-of-vocabulary rate are reported alongside as secondary, and are labelled as such
because they were not the pre-specified one.

Significance is a permutation test on the pairing, which assumes nothing about the
distribution of either variable — per-query nDCG differences are mostly exactly zero,
which no parametric test handles gracefully.
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
from groundwork.eval import correlation_permutation_test, holm_bonferroni
from groundwork.retrieval import LUCENE_ENGLISH_STOPWORDS, BM25Retriever, Tokenizer

PRIMARY_PREDICTOR = "max_idf"
SECONDARY_PREDICTORS = ("mean_idf", "out_of_vocabulary_rate", "num_terms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lexical", required=True, help="Results file for the lexical system")
    parser.add_argument("--semantic", required=True, help="Results file for the semantic system")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_per_query(path: Path, metric: str) -> dict[str, float]:
    """Per-query scores for ``metric`` from a results file."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    per_query_path = path.parent / summary["per_query_file"]
    per_query = json.loads(per_query_path.read_text(encoding="utf-8"))["per_query"]
    return {qid: float(scores[metric]) for qid, scores in per_query.items()}


def main() -> int:
    args = parse_args()

    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    lexical = load_per_query(Path(args.lexical), args.metric)
    semantic = load_per_query(Path(args.semantic), args.metric)

    if lexical.keys() != semantic.keys():
        raise ValueError("the two systems were scored on different query sets")

    tokenizer = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)
    index = BM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)
    index.index(dataset.corpus)

    query_ids = sorted(lexical)
    stats = {qid: index.query_statistics(dataset.queries[qid]) for qid in query_ids}
    advantage = [lexical[qid] - semantic[qid] for qid in query_ids]

    print(f"{args.dataset} / {args.split} / {args.metric}")
    print(f"  lexical  {Path(args.lexical).name}")
    print(f"  semantic {Path(args.semantic).name}")
    print(f"  {len(query_ids)} queries, mean advantage {np.mean(advantage):+.4f}")
    print()

    predictors = (PRIMARY_PREDICTOR, *SECONDARY_PREDICTORS)
    results = {}
    for name in predictors:
        values = [stats[qid][name] for qid in query_ids]
        outcome = correlation_permutation_test(
            values, advantage, num_permutations=args.permutations, seed=args.seed
        )
        results[name] = outcome.describe()
        role = "PRIMARY  " if name == PRIMARY_PREDICTOR else "secondary"
        print(f"  {role} {name:<24} rho {outcome.correlation:+.4f}   p {outcome.p_value:.4f}")

    # Secondary predictors are a family; the primary one was pre-specified and is not
    # adjusted, because adjusting a hypothesis fixed in advance for tests invented
    # afterwards would penalise it for the exploration rather than the other way round.
    secondary_names = list(SECONDARY_PREDICTORS)
    adjusted = holm_bonferroni([results[name]["p_value"] for name in secondary_names])
    for name, value in zip(secondary_names, adjusted, strict=True):
        results[name]["p_value_holm_secondary"] = value

    primary_values = [stats[qid][PRIMARY_PREDICTOR] for qid in query_ids]
    order = np.argsort(primary_values)
    thirds = np.array_split(order, 3)
    print(f"\n  advantage by {PRIMARY_PREDICTOR} tercile (descriptive only):")
    tercile_summary = []
    for label, chunk in zip(("lowest", "middle", "highest"), thirds, strict=True):
        chunk_advantage = [advantage[i] for i in chunk]
        chunk_values = [primary_values[i] for i in chunk]
        entry = {
            "tercile": label,
            "num_queries": len(chunk),
            f"{PRIMARY_PREDICTOR}_range": [min(chunk_values), max(chunk_values)],
            "mean_advantage": float(np.mean(chunk_advantage)),
        }
        tercile_summary.append(entry)
        print(
            f"    {label:<8} n={len(chunk):<4} {PRIMARY_PREDICTOR} "
            f"{min(chunk_values):.2f}-{max(chunk_values):.2f}   "
            f"mean advantage {np.mean(chunk_advantage):+.4f}"
        )

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "metric": args.metric,
        "lexical_run": args.lexical,
        "semantic_run": args.semantic,
        "num_queries": len(query_ids),
        "mean_advantage": float(np.mean(advantage)),
        "hypothesis": (
            f"per-query ({args.metric} lexical - semantic) correlates positively with "
            f"{PRIMARY_PREDICTOR}"
        ),
        "primary_predictor": PRIMARY_PREDICTOR,
        "correlations": results,
        "terciles": tercile_summary,
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    output = (
        Path(args.output)
        if args.output
        else Path("results") / f"{args.dataset}-query-analysis.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
