"""Score an LLM relevance judge against the humans who wrote the qrels.

Usage::

    python scripts/run_judge.py --dataset scidocs
    python scripts/run_judge.py --dataset trec-covid --max-pairs 4000

Experiment 16. "LLM-as-judge" is how RAG systems are usually evaluated now, and the
judge's verdict is usually reported as ground truth. This asks what it is worth, using the
one thing this benchmark has that a typical RAG evaluation does not: thousands of
relevance decisions made by people.

Agreement is Cohen's kappa rather than raw agreement, because relevance pools are
overwhelmingly non-relevant and a judge that always says no scores above 90% raw agreement
while discriminating nothing.

Only datasets whose qrels contain explicit non-relevant judgements can be used. SciFact
and NFCorpus ship only positive ones, so there is no human "no" to agree with, and the
loader refuses them rather than reporting a kappa computed against a convention.
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
from groundwork.data import BeirDataset, load_beir_dataset
from groundwork.eval.judge import DEFAULT_JUDGE, RelevanceJudge, calibrate, judgeable_pairs
from groundwork.retrieval import (
    LUCENE_ENGLISH_STOPWORDS,
    MultiFieldBM25Retriever,
    Tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scidocs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--depth", type=int, default=10, help="How far down each ranking to judge")
    parser.add_argument("--model", default=DEFAULT_JUDGE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Judge at most this many pairs, sampled deterministically",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def retrieve_pool(
    dataset: BeirDataset, top_k: int
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    """Retrieve the pool of documents the judge will be asked about.

    Returns ``(run, retriever_settings)``.

    Multi-field BM25, the repository default. The judge is being calibrated, not the
    retriever, so the pool only has to be a realistic one — the documents a RAG system
    would actually put in front of a model. BM25 is used rather than the best available
    system because it needs no optional extras, which keeps this experiment reproducible
    for anyone who did not install torch.
    """
    tokenizer = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)
    retriever = MultiFieldBM25Retriever(tokenizer=tokenizer)
    retriever.index(dataset.corpus)
    return retriever.retrieve(dataset.queries, top_k=top_k), retriever.describe()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    run, retriever_settings = retrieve_pool(dataset, top_k=max(args.depth, 100))

    # Raises when the qrels carry no non-relevant judgements, which is the case for
    # SciFact and NFCorpus and is the reason this experiment covers two datasets.
    pairs = judgeable_pairs(dataset.qrels, run, depth=args.depth)
    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        step = len(pairs) / args.max_pairs
        pairs = [pairs[int(i * step)] for i in range(args.max_pairs)]

    print(f"{args.dataset} / {args.split} / judging with {args.model}")
    print(f"  {len(pairs)} judged pairs in the top {args.depth} of a BM25 ranking")
    relevant = sum(1 for _, _, grade in pairs if grade > 0)
    print(f"  humans called {relevant} of them relevant ({100 * relevant / len(pairs):.1f}%)")

    judge = RelevanceJudge(model_name=args.model, batch_size=args.batch_size, seed=args.seed)
    texts = [
        (
            dataset.queries[query_id],
            f"{dataset.corpus[doc_id].get('title', '')} {dataset.corpus[doc_id].get('text', '')}",
        )
        for query_id, doc_id, _ in pairs
    ]
    start = time.perf_counter()
    verdicts = judge.judge(texts)
    seconds = time.perf_counter() - start

    human = [grade for _, _, grade in pairs]
    agreement, unparseable = calibrate(human, verdicts)

    said_yes = sum(1 for v in verdicts if v == 1)
    print(f"\n  judged in {seconds:.0f}s, {unparseable} unparseable")
    print(f"  the judge said yes {said_yes} times ({100 * said_yes / len(verdicts):.1f}%)")
    print("-" * 62)
    print(f"  raw agreement   {agreement.raw_agreement:.4f}")
    print(f"  chance          {agreement.chance_agreement:.4f}")
    print(f"  Cohen's kappa   {agreement.cohens_kappa:.4f}")
    print(f"  confusion       {agreement.confusion}  (labels {agreement.labels})")

    # Which way the judge is wrong matters more than how often: calling non-relevant
    # documents relevant and missing relevant ones have different consequences.
    matrix = np.array(agreement.confusion)
    false_positive = int(matrix[0][1]) if matrix.shape == (2, 2) else None
    false_negative = int(matrix[1][0]) if matrix.shape == (2, 2) else None
    if false_positive is not None:
        print(f"  called relevant when the human did not: {false_positive}")
        print(f"  missed what the human called relevant:  {false_negative}")

    suffix = f"-{args.tag}" if args.tag else ""
    output = (
        Path(args.output) if args.output else Path("results") / f"{args.dataset}-judge{suffix}.json"
    )
    record = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "method": "llm-judge-calibration",
        "pool_retriever": retriever_settings,
        "depth": args.depth,
        "num_pairs": len(pairs),
        "num_human_relevant": relevant,
        "num_judge_yes": said_yes,
        "num_unparseable": unparseable,
        "agreement": agreement.describe(),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "judge": judge.describe(),
        "timing_seconds": {"judge": round(seconds, 1)},
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
