"""Measure what generated expansions contribute, before any retrieval happens.

Usage::

    python scripts/analyse_expansions.py --dataset nfcorpus --kind query --split test
    python scripts/analyse_expansions.py --dataset nfcorpus --kind document

Experiments 13 and 14 compare generative expansion against RM3, which adds a *tunable
number* of feedback terms — 50 on both datasets here. A generative expander adds whatever
the model happened to write. If those differ by an order of magnitude then a result like
"HyDE does not beat RM3" is partly a statement about expansion length rather than about
where the terms came from, and the write-up has to say so.

This script exists so that figure is a measurement rather than a sentence. It loads
committed expansions, tokenises them the way the index does, and records how many terms
each expansion actually adds that the original did not already contain. Terms already in
the original are excluded: repeating them changes term frequency, not which documents can
be reached at all.

No model is loaded and no retrieval is run, so this is seconds rather than an hour.
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
from groundwork.retrieval import LUCENE_ENGLISH_STOPWORDS, Tokenizer
from groundwork.retrieval.expansion import summarise_expansions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--kind", choices=["query", "document"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--expansions", default=None)
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def originals(dataset: object, kind: str) -> dict[str, str]:
    """The text each expansion is appended to, as one string per id."""
    if kind == "query":
        return dict(dataset.queries)
    return {
        doc_id: f"{fields.get('title', '')} {fields.get('text', '')}".strip()
        for doc_id, fields in dataset.corpus.items()
    }


def main() -> int:
    args = parse_args()

    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    # A corpus has no split, so document expansions are generated once under "test".
    default_split = args.split if args.kind == "query" else "test"
    path = (
        Path(args.expansions)
        if args.expansions
        else Path(args.data_dir) / "expansions" / f"{args.dataset}-{args.kind}-{default_split}.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run generate_expansions.py --dataset {args.dataset} "
            f"--kind {args.kind} first."
        )

    record = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer(
        stopwords=None if args.no_stopwords else LUCENE_ENGLISH_STOPWORDS,
        stem=not args.no_stem,
    )
    summary = summarise_expansions(originals(dataset, args.kind), record["expansions"], tokenizer)

    print(f"{args.dataset} / {args.split} / {args.kind} expansion")
    print(f"  from {path.name} ({record['model']}, {record['generations_per_input']} per input)")
    print("-" * 62)
    print(f"  inputs                    {summary.num_inputs}")
    print(f"  generated nothing         {summary.num_empty}")
    print(f"  added no new terms        {summary.num_adding_nothing}")
    words, added = summary.generated_words, summary.added_terms
    print(f"  generated words           median {words['median']:.0f}  mean {words['mean']:.1f}")
    print(f"                            p25 {words['p25']:.0f}  p75 {words['p75']:.0f}")
    print(f"  NEW indexable terms       median {added['median']:.0f}  mean {added['mean']:.1f}")
    print(f"                            p25 {added['p25']:.0f}  p75 {added['p75']:.0f}")
    print(f"  fraction of generated     {summary.added_term_fraction:.4f}")
    print("    vocabulary that is new")

    suffix = f"-{default_split}" if args.kind == "query" else ""
    output = (
        Path(args.output)
        if args.output
        else Path("results") / f"{args.dataset}-{args.kind}-expansion-stats{suffix}.json"
    )
    out = {
        "dataset": args.dataset,
        "split": default_split,
        "kind": args.kind,
        "method": "expansion-vocabulary-analysis",
        "expansion": {k: v for k, v in record.items() if k != "expansions"},
        "tokenizer": tokenizer.describe(),
        "statistics": summary.describe(),
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
