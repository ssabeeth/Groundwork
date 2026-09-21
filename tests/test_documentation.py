"""The README's numbers must be numbers this repository produced.

The project's first rule is that every claim in the README is backed by a run, and that
no figure anywhere is estimated or invented. Until now that was enforced by remembering,
which is exactly the invariant that rots.

Two checks, and the second is the one that matters.

The first pins named headline figures to specific results files. It is useful but weak
on its own: it asks whether a number appears *somewhere* in the README, so if the same
value occurs in two tables, corrupting one of them still passes.

The second inverts the question. It extracts every three- and four-decimal figure in the
README and requires each to be traceable to a value in `results/` — as a metric, a
derived delta, or a percentage of one. That is the check that catches a fabricated
number, because a figure nothing produced has nowhere to come from. Finding an
untraceable figure means either the prose invented it, or a real computation is not
being written to `results/` and should be.

Numbers that are legitimately not measurements — version numbers, rank cutoffs, worked
arithmetic in prose — are listed in NON_MEASUREMENT with a reason each.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"

# (results file, dotted path into the JSON, format spec)
CLAIMS = [
    ("scifact-bm25.json", "metrics.ndcg@10", ".4f"),
    ("scifact-bm25.json", "metrics.recall@100", ".4f"),
    ("nfcorpus-bm25-graded.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-bm25-graded.json", "metrics.recall@100", ".4f"),
    ("trec-covid-bm25-graded.json", "metrics.ndcg@10", ".4f"),
    ("scifact-dense.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-dense.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-dense.json", "metrics.recall@100", ".4f"),
    ("scifact-rm3-tuned.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-rm3-tuned.json", "metrics.ndcg@10", ".4f"),
    ("scifact-rrf-tuned.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-rrf-tuned.json", "metrics.ndcg@10", ".4f"),
    ("scifact-rrf-rerank-reranked.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-rrf-rerank-reranked.json", "metrics.ndcg@10", ".4f"),
]


def dig(record: dict, dotted: str):
    """Fetch a value by dotted path, so metric names containing '@' still work."""
    value = record
    for key in dotted.split("."):
        value = value[key]
    return value


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize(("filename", "path", "spec"), CLAIMS)
def test_readme_quotes_the_recorded_number(readme, filename, path, spec):
    record = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    formatted = format(dig(record, path), spec)
    assert formatted in readme, (
        f"README does not contain {formatted}, which is {path} in results/{filename}. "
        "Either a re-run moved the number and the prose is stale, or the prose was "
        "never backed by a run."
    )


def test_every_committed_result_has_its_settings_recorded():
    """A score without the settings that produced it is not reproducible."""
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in record:
            continue  # sweep and comparison files have their own shapes
        assert "groundwork_version" in record, f"{path.name} has no version"
        assert record.get("retriever") or record.get("retrievers"), (
            f"{path.name} records metrics but not what produced them"
        )


def test_graded_results_record_their_gain_function():
    """On graded qrels the gain choice moves nDCG, so it must never be implicit."""
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("graded_qrels"):
            assert record.get("gain") in {"exponential", "linear"}, (
                f"{path.name} is graded but does not say which gain function was used"
            )
            assert "metrics_by_gain" in record, f"{path.name} does not record both gains"


def test_per_query_files_exist_for_every_run_that_points_at_one():
    """A paired comparison must be reproducible from a clean clone."""
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        pointer = record.get("per_query_file")
        if pointer:
            assert (path.parent / pointer).exists(), (
                f"{path.name} points at {pointer}, which is not committed"
            )


# Figures in the README that are not measurements and so have no results file. Each one
# needs a reason, because "add it to the allowlist" is how this check would be defanged.
NON_MEASUREMENT: dict[str, str] = {
    "0.665": "BEIR's published SciFact BM25 figure, quoted from the paper",
    "0.656": "BEIR's published TREC-COVID BM25 figure, quoted from the paper",
    "0.325": "BEIR's published NFCorpus BM25 figure, quoted from the paper",
    "0.001": "a threshold in prose ('p < 0.001'), not a measured value",
    "0.0001": "a threshold in prose ('p < 0.0001'), not a measured value",
}


def _numeric_values(node: object):
    """Every number anywhere in a parsed results file."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _numeric_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _numeric_values(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield float(node)


@pytest.fixture(scope="module")
def traceable_figures() -> set[str]:
    """Every value in results/, in each string form the README might quote it as."""
    figures: set[str] = set()
    for path in RESULTS.rglob("*.json"):
        for value in _numeric_values(json.loads(path.read_text(encoding="utf-8"))):
            magnitude = abs(value)
            for spec in (".2f", ".3f", ".4f"):
                figures.add(format(magnitude, spec))
            figures.add(format(magnitude * 100, ".1f"))
    return figures


def test_no_figure_in_the_readme_is_untraceable(readme, traceable_figures):
    """Every 3-4 decimal figure must come from a run, not from prose.

    This is the check that enforces "no fabricated or estimated figures anywhere". If it
    fails, either a number was invented, or a genuine computation is being done outside
    the repository and needs a script that writes its result to results/.
    """
    quoted = set(re.findall(r"\d+\.\d{3,4}", readme))
    untraceable = sorted(
        figure
        for figure in quoted
        if figure not in traceable_figures and figure not in NON_MEASUREMENT
    )
    assert not untraceable, (
        f"README quotes figures no results file contains: {untraceable}. "
        "Either the number was not produced by a run, or the script that produces it "
        "does not write to results/."
    )


@pytest.mark.parametrize("document", ["docs/experiments.md", "docs/decisions.md"])
def test_no_figure_in_the_docs_is_untraceable(traceable_figures, document):
    """The experiment log is the primary record, so it is held to the same rule.

    A number in the log that no results file contains is either invented or comes from a
    computation living outside the repository. Both are the same bug from a reader's
    point of view: the claim cannot be checked.
    """
    text = (ROOT / document).read_text(encoding="utf-8")
    quoted = set(re.findall(r"\d+\.\d{3,4}", text))
    untraceable = sorted(
        figure
        for figure in quoted
        if figure not in traceable_figures and figure not in NON_MEASUREMENT
    )
    assert not untraceable, f"{document} quotes figures no results file contains: {untraceable}."
