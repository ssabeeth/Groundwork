"""Run the BM25 baseline on a BEIR dataset and record the result.

Usage::

    python scripts/run_baseline.py --dataset scifact

Every number in the README comes from this script. The settings that produced a result
are written into the results file alongside it, because a retrieval score without its
tokenisation and parameters is not reproducible.
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

from groundwork import __version__
from groundwork.data import load_beir_dataset
from groundwork.eval import evaluate_run
from groundwork.retrieval import BM25Retriever, Tokenizer

# BM25 nDCG@10 as published in the BEIR paper (Thakur et al., 2021), which used
# Elasticsearch with k1=0.9, b=0.4. A reimplementation will not match to three decimal
# places - tokenisation and stemming differ - but landing far outside the tolerance
# below means the harness is wrong, not the method.
REFERENCE_NDCG_10 = {
    "scifact": 0.665,
    "trec-covid": 0.656,
    "nfcorpus": 0.325,
}
REFERENCE_TOLERANCE = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--split", default="test", help="Qrels split to score")
    parser.add_argument("--data-dir", default="data", help="Where datasets are cached")
    parser.add_argument("--k1", type=float, default=0.9, help="BM25 k1")
    parser.add_argument("--b", type=float, default=0.4, help="BM25 b")
    parser.add_argument("--top-k", type=int, default=100, help="Documents retrieved per query")
    parser.add_argument("--no-stem", action="store_true", help="Disable Porter stemming")
    parser.add_argument("--no-stopwords", action="store_true", help="Keep stopwords")
    parser.add_argument(
        "--output",
        default=None,
        help="Results JSON path (default: results/<dataset>-bm25.json)",
    )
    parser.add_argument("--tag", default="", help="Short label for this run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading {args.dataset} ({args.split})")
    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    print(f"  {len(dataset.corpus)} documents, {len(dataset.queries)} judged queries")

    judgements = sum(len(d) for d in dataset.qrels.values())
    relevant = sum(1 for d in dataset.qrels.values() for level in d.values() if level > 0)
    levels = sorted({level for d in dataset.qrels.values() for level in d.values()})
    print(f"  {judgements} judgements, {relevant} relevant, levels {levels}")

    tokenizer = Tokenizer(
        stopwords=None if args.no_stopwords else Tokenizer().stopwords,
        stem=not args.no_stem,
    )
    retriever = BM25Retriever(k1=args.k1, b=args.b, tokenizer=tokenizer)

    start = time.perf_counter()
    retriever.index(dataset.corpus)
    index_seconds = time.perf_counter() - start

    start = time.perf_counter()
    run = retriever.retrieve(dataset.queries, top_k=args.top_k)
    retrieve_seconds = time.perf_counter() - start

    metrics = evaluate_run(run, dataset.qrels, k_values=(1, 10, 100))

    print(f"\n{args.dataset} / {args.split} / BM25 (k1={args.k1}, b={args.b})")
    print("-" * 52)
    for name in sorted(metrics):
        if name != "num_queries":
            print(f"  {name:<14} {metrics[name]:.4f}")
    print(f"  {'queries':<14} {int(metrics['num_queries'])}")
    print(f"  {'index time':<14} {index_seconds:.1f}s")
    print(f"  {'retrieve time':<14} {retrieve_seconds:.1f}s")

    reference = REFERENCE_NDCG_10.get(args.dataset)
    within_tolerance = None
    if reference is not None:
        delta = metrics["ndcg@10"] - reference
        within_tolerance = abs(delta) <= REFERENCE_TOLERANCE
        verdict = "ok" if within_tolerance else "OUT OF RANGE - check tokenisation"
        print(f"\n  BEIR published BM25 nDCG@10: {reference:.3f}")
        print(f"  This run:                    {metrics['ndcg@10']:.4f}  ({delta:+.4f})  {verdict}")

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "metrics": {k: v for k, v in metrics.items() if k != "num_queries"},
        "num_queries": int(metrics["num_queries"]),
        "num_documents": len(dataset.corpus),
        "retriever": retriever.describe(),
        "top_k": args.top_k,
        "timing_seconds": {
            "index": round(index_seconds, 2),
            "retrieve": round(retrieve_seconds, 2),
        },
        "reference_ndcg_at_10": reference,
        "within_reference_tolerance": within_tolerance,
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    output = Path(args.output) if args.output else Path("results") / f"{args.dataset}-bm25.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")

    return 0 if within_tolerance is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
