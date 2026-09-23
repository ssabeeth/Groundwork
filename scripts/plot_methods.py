"""Draw the README's headline chart from committed results.

    python scripts/plot_methods.py

Reads nDCG@10 for every method compared on SciFact and NFCorpus (Annexe A.11 and A.13)
and writes docs/figures/ndcg10-by-method.svg. `tests/test_figures.py` regenerates the
figure and fails if the committed file differs, so it cannot drift from the results.
"""

import argparse
import json
from pathlib import Path

from groundwork.figures import Group, Row, dot_plot_svg

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "docs" / "figures" / "ndcg10-by-method.svg"

DATASETS = {"SciFact": "scifact", "NFCorpus": "nfcorpus"}

FUSION = "Fusion of keyword + another retriever"
GENERATED = "Adds LLM-generated text"
OTHER = "Other methods"
GROUPS = [
    Group(FUSION, light="#2a78d6", dark="#3987e5"),
    Group(GENERATED, light="#eb6834", dark="#d95926"),
    Group(OTHER, light="#8a8984", dark="#8a8984"),
]

# (label, group, results file with {dataset} in place of the dataset name)
METHODS = [
    ("Fusion: BM25 + dense (bge-small)", FUSION, "{dataset}-rrf-bge.json"),
    ("SPLADE (learned sparse)", OTHER, "{dataset}-splade-len512.json"),
    ("Dense embeddings (bge-small)", OTHER, "{dataset}-dense-bge.json"),
    ("Fusion: BM25 + ColBERT", FUSION, "{dataset}-rrf-colbertfusion.json"),
    ("ColBERT (late interaction)", OTHER, "{dataset}-colbert-len512.json"),
    ("Fusion + MiniLM reranker", OTHER, "{dataset}-rrf-rerank-bge.json"),
    ("Fusion + bge reranker", OTHER, "{dataset}-rrf-rerank-bgererank.json"),
    ("RM3 (keyword feedback)", OTHER, "{dataset}-rm3-tuned.json"),
    (
        "doc2query (generated document text)",
        GENERATED,
        "{dataset}-document-expansion-doc2query.json",
    ),
    ("LLM query expansion (HyDE)", GENERATED, "{dataset}-query-expansion-hyde.json"),
]
BASELINE = (
    "BM25 keyword baseline",
    {"scifact": "scifact-bm25.json", "nfcorpus": "nfcorpus-bm25-graded.json"},
)


def _ndcg10(filename: str) -> float:
    record = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    return record["metrics"]["ndcg@10"]


def build() -> str:
    """The figure's SVG, computed from results/ alone."""
    reference = Row(BASELINE[0], OTHER, tuple(_ndcg10(BASELINE[1][d]) for d in DATASETS.values()))
    rows = [
        Row(
            label,
            group,
            tuple(_ndcg10(pattern.format(dataset=d)) for d in DATASETS.values()),
            annotate=(i == 0),
        )
        for i, (label, group, pattern) in enumerate(METHODS)
    ]
    # Order by average gain over BM25, so the ranking is read top to bottom.
    rows.sort(key=lambda r: -sum(v - b for v, b in zip(r.values, reference.values, strict=True)))
    return dot_plot_svg(
        rows,
        panels=list(DATASETS),
        groups=GROUPS,
        reference=reference,
        title="Fusing keyword and dense search scored highest on both datasets",
        subtitle=(
            "nDCG@10 on each BEIR test set; higher is better. Every method ran on the same harness."
        ),
        caption=(
            "Values from results/, drawn by scripts/plot_methods.py. "
            "Full tables: README Annexe A.11 and A.13."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")
    print(f"wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
