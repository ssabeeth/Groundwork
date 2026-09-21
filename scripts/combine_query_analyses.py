"""Collect the query-dependence analyses and correct across the family.

Usage::

    python scripts/combine_query_analyses.py \
        --analysis results/scifact-query-analysis.json \
        --analysis results/scifact-query-analysis-bge.json \
        --analysis results/nfcorpus-query-analysis.json \
        --analysis results/nfcorpus-query-analysis-bge.json

`analyse_queries.py` tests one dataset against one encoder and Holm-corrects only its
*secondary* predictors, because the primary hypothesis was fixed in advance and adjusting
it for tests invented afterwards would penalise it for the exploration.

But the primary hypothesis is now asked several times over — once per dataset and encoder
— and that is a family too. Asking one question of four query sets and reporting the
smallest p-value is the same error the secondary correction exists to prevent, one level
up.

This existed as four numbers in a README table before it existed as code, worked out by
hand. That is precisely the thing `tests/test_documentation.py` is supposed to make
impossible, and it slipped through because the traceability check's haystack was large
enough that the hand-computed values collided with unrelated figures. Both the check and
the gap it missed are recorded in the experiment log.
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
from groundwork.eval import holm_bonferroni


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        metavar="FILE",
        help="A query-analysis results file; repeat for each member of the family",
    )
    parser.add_argument(
        "--output",
        default="results/query-dependence-summary.json",
        help="Where to write the corrected family",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyses = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.analysis]

    predictors = {analysis["primary_predictor"] for analysis in analyses}
    if len(predictors) != 1:
        raise ValueError(
            f"these analyses do not share one primary predictor ({sorted(predictors)}), "
            "so they are not one family and correcting across them would be meaningless"
        )
    predictor = predictors.pop()

    members = []
    for analysis, source in zip(analyses, args.analysis, strict=True):
        correlation = analysis["correlations"][predictor]
        members.append(
            {
                "source": source,
                "dataset": analysis["dataset"],
                "semantic_run": analysis["semantic_run"],
                "num_queries": analysis["num_queries"],
                "mean_advantage": analysis["mean_advantage"],
                "correlation": correlation["correlation"],
                "p_value": correlation["p_value"],
            }
        )

    adjusted = holm_bonferroni([member["p_value"] for member in members])
    for member, value in zip(members, adjusted, strict=True):
        member["p_value_holm"] = value

    correlations = [member["correlation"] for member in members]
    summary = {
        "predictor": predictor,
        "family_size": len(members),
        "members": members,
        "all_positive": all(value > 0 for value in correlations),
        "num_significant_after_holm": sum(1 for m in members if m["p_value_holm"] < 0.05),
        "min_correlation": min(correlations),
        "max_correlation": max(correlations),
        "median_correlation": float(np.median(correlations)),
        "groundwork_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    print(f"Primary predictor {predictor}, Holm-corrected across {len(members)} tests")
    print("-" * 74)
    print(f"  {'dataset':<12} {'encoder':<28} {'rho':>8} {'p':>9} {'Holm':>9}")
    for member in members:
        encoder = Path(member["semantic_run"]).stem
        print(
            f"  {member['dataset']:<12} {encoder:<28} {member['correlation']:+8.4f} "
            f"{member['p_value']:9.4f} {member['p_value_holm']:9.4f}"
        )
    print(
        f"\n  all positive: {summary['all_positive']}   "
        f"significant after Holm: {summary['num_significant_after_holm']}/{len(members)}   "
        f"median rho {summary['median_correlation']:+.4f}"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
