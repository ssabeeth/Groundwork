"""Generate query or document expansions once, and commit them.

Usage::

    python scripts/generate_expansions.py --dataset nfcorpus --kind query
    python scripts/generate_expansions.py --dataset nfcorpus --kind document --num 5

Generation is the slow, model-dependent, non-deterministic part of experiments 13 and 14,
so it is separated from retrieval entirely. This script writes text to
``data/expansions/`` and nothing else; ``run_expanded.py`` reads that text and scores it.

The separation is what makes the experiments reproducible. Re-scoring an expanded run —
sweeping the query weight, trying a different fusion, re-running after a metric fix —
must not require re-running a language model over 25,000 documents, and must not produce
slightly different text when it does.

Expansions are written to ``data/`` rather than ``results/`` because they are inputs, not
measurements, and because doc2query output for a large corpus is far too big to commit.
The generation settings that would change the text are recorded alongside it.
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
from groundwork.retrieval.expansion import (
    DEFAULT_DOCUMENT_EXPANDER,
    DEFAULT_QUERY_EXPANDER,
    HYDE_PROMPT,
    generate_expansions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--kind", choices=["query", "document"], required=True)
    parser.add_argument("--model", default=None, help="Defaults to the expander for --kind")
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Generations per input; defaults to 1 for queries and 5 for documents",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Generate for the first N only")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = load_beir_dataset(args.dataset, split=args.split, data_dir=args.data_dir)

    if args.kind == "query":
        model_name = args.model or DEFAULT_QUERY_EXPANDER
        texts = dict(dataset.queries)
        prompt = HYDE_PROMPT
        num = args.num if args.num is not None else 1
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 128
    else:
        model_name = args.model or DEFAULT_DOCUMENT_EXPANDER
        # doc2query is trained on passage text, not on a title/body pair, so the document
        # is presented the way the model saw its training data.
        texts = {
            doc_id: f"{fields.get('title', '')} {fields.get('text', '')}".strip()
            for doc_id, fields in dataset.corpus.items()
        }
        prompt = None
        num = args.num if args.num is not None else 5
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 64

    if args.limit is not None:
        texts = dict(list(texts.items())[: args.limit])

    # Several generations only differ if decoding samples; the library call enforces this
    # too, but failing here saves an hour of generation before the error.
    do_sample = num > 1

    print(f"{args.dataset} / {args.split} / {args.kind} expansion")
    print(f"  {len(texts)} inputs, {num} generation(s) each, model {model_name}")

    start = time.perf_counter()
    generated = generate_expansions(
        texts,
        model_name=model_name,
        prompt=prompt,
        num_return_sequences=num,
        max_new_tokens=max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        do_sample=do_sample,
    )
    seconds = time.perf_counter() - start

    empty = sum(1 for values in generated.values() if not any(v.strip() for v in values))
    print(f"  generated in {seconds:.0f}s, {empty} input(s) produced nothing")
    for example_id in list(generated)[:2]:
        preview = generated[example_id][0][:160].replace("\n", " ")
        print(f"    {example_id}: {preview}")

    record = {
        "dataset": args.dataset,
        "split": args.split,
        "kind": args.kind,
        "model": model_name,
        "prompt": prompt,
        "generations_per_input": num,
        "sampling": do_sample,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "num_inputs": len(texts),
        "num_empty": empty,
        "seconds": round(seconds, 1),
        "expansions": generated,
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    output = (
        Path(args.output)
        if args.output
        else Path(args.data_dir) / "expansions" / f"{args.dataset}-{args.kind}-{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
