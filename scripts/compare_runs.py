"""Compare retrieval runs with a paired significance test.

Usage::

    python scripts/compare_runs.py \
        --pair results/scifact-bm25.json results/scifact-bm25-no-stem.json \
        --pair results/scifact-bm25-no-stopwords.json results/scifact-bm25-plain.json

Each ``--pair`` names two results files written by ``run_baseline.py``. The per-query
scores they point at are loaded, aligned by query id, and put through a paired
randomisation test. When more than one pair is given the p-values are additionally
adjusted with Holm-Bonferroni, because asking several questions of one query set makes
it easier to find a "significant" answer by chance.

A difference is reported with both its size and its p-value on purpose: the p-value
says whether the difference is distinguishable from noise, and only the effect size
says whether it is worth anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from groundwork.eval import holm_bonferroni, paired_test_from_per_query

DEFAULT_METRIC = "ndcg@10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        required=True,
        help="Two results JSON files to compare; repeat for several comparisons",
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric to test")
    parser.add_argument(
        "--resamples",
        type=int,
        default=10_000,
        help="Sign assignments to sample when the query set is too large to enumerate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the sampled test")
    parser.add_argument("--output", default=None, help="Write comparisons to this JSON file")
    return parser.parse_args()


def load_per_query(summary_path: Path, metric: str) -> tuple[str, dict[str, float]]:
    """Load one system's per-query scores for ``metric``.

    Args:
        summary_path: A results file written by ``run_baseline.py``.
        metric: Metric name, e.g. ``"ndcg@10"``.

    Returns:
        ``(label, {query_id: score})``, where the label identifies the configuration.

    Raises:
        FileNotFoundError: If the summary or its per-query file is missing.
        KeyError: If the summary predates per-query output, or lacks the metric.
    """
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if "per_query_file" not in summary:
        raise KeyError(
            f"{summary_path} has no per_query_file: it was written before per-query "
            f"scores were recorded. Re-run run_baseline.py to regenerate it."
        )

    per_query_path = summary_path.parent / summary["per_query_file"]
    if not per_query_path.exists():
        raise FileNotFoundError(f"{summary_path} points at {per_query_path}, which is missing")

    per_query = json.loads(per_query_path.read_text(encoding="utf-8"))["per_query"]

    scores = {}
    for query_id, metrics in per_query.items():
        if metric not in metrics:
            raise KeyError(f"{per_query_path} has no {metric!r}; found {sorted(metrics)}")
        scores[query_id] = float(metrics[metric])

    return describe_run(summary), scores


def describe_run(summary: dict) -> str:
    """A short label naming the system and the settings that distinguish it.

    Built defensively across methods: a dense run has no tokeniser and a fused run has
    no single set of parameters, so anything method-specific is included only when it
    is actually there.

    Args:
        summary: A parsed results file.

    Returns:
        A one-line label.
    """
    retriever = summary.get("retriever", {})
    method = summary.get("method") or retriever.get("method") or "bm25"
    parts: list[str] = []

    if method == "dense":
        parts.append(str(retriever.get("model", "?")))
        if retriever.get("truncation_rate") is not None:
            parts.append(f"truncated={100 * retriever['truncation_rate']:.0f}%")
    elif method == "rrf":
        parts.append("+".join(summary.get("systems", [])))
        parts.append(f"k={summary.get('rrf_k')}")
    else:
        if method == "rm3":
            parts.append(
                f"fb={retriever.get('fb_docs')}, terms={retriever.get('fb_terms')}, "
                f"alpha={retriever.get('alpha')}"
            )
        tokenizer = retriever.get("tokenizer") or {}
        if tokenizer.get("stem"):
            parts.append(f"stem={tokenizer['stem']}, stopwords={tokenizer['stopwords']}")

    tag = summary.get("tag") or method
    detail = ", ".join(part for part in parts if part)
    return f"{tag} [{method}]" + (f" ({detail})" if detail else "")


def main() -> int:
    args = parse_args()

    comparisons = []
    for a_path, b_path in args.pair:
        a_label, a_scores = load_per_query(Path(a_path), args.metric)
        b_label, b_scores = load_per_query(Path(b_path), args.metric)
        result = paired_test_from_per_query(
            a_scores, b_scores, num_resamples=args.resamples, seed=args.seed
        )
        comparisons.append(
            {
                "a": {"file": a_path, "label": a_label},
                "b": {"file": b_path, "label": b_label},
                "metric": args.metric,
                **result.describe(),
            }
        )

    if len(comparisons) > 1:
        adjusted = holm_bonferroni([c["p_value"] for c in comparisons])
        for comparison, value in zip(comparisons, adjusted, strict=True):
            comparison["p_value_holm"] = value

    print(f"Paired randomisation test on {args.metric}")
    print("=" * 78)
    for comparison in comparisons:
        print(f"\n  {comparison['a']['label']}")
        print(f"  vs {comparison['b']['label']}")
        print(f"    delta         {comparison['delta']:+.4f}")
        print(f"    p             {comparison['p_value']:.4f}")
        if "p_value_holm" in comparison:
            print(f"    p (Holm)      {comparison['p_value_holm']:.4f}")
        print(f"    queries       {comparison['num_queries']} ({comparison['num_ties']} tied)")
        print(f"    method        {comparison['method']} over {comparison['num_assignments']}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "metric": args.metric,
            "seed": args.seed,
            "resamples": args.resamples,
            "holm_adjusted": len(comparisons) > 1,
            "comparisons": comparisons,
        }
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
