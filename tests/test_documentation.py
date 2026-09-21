"""The README's numbers must be numbers this repository produced.

The project's first rule is that every claim in the README is backed by a run, and that
no figure anywhere is estimated or invented. Until now that was enforced by remembering,
which is exactly the invariant that rots.

Four checks, and they have to hold together rather than individually.

**Pinned claims** tie named headline figures to specific results files. Useful but weak
alone: it asks whether a number appears *somewhere* in the README, so if the same value
occurs in two tables, corrupting one still passes.

**Traceability** inverts the question. Every three- and four-decimal figure in the README
and in `docs/` must match a value in `results/`. This is the check that catches an
invented number, because a figure nothing produced has nowhere to come from.

Its strength is entirely a function of how small the haystack is, and that was measured
rather than assumed after it let one through. Drawing from *every* number in `results/`,
including per-query blocks, the traceable set covered 96% of all possible three-decimal
values in [0, 1): a fabricated figure passed by collision almost every time, and one had
— the Holm column of the query-dependence table, computed by hand, matching an unrelated
fusion delta. Restricted to summary values it covers 36% of three-decimal and 5% of
four-decimal strings. `test_the_traceable_set_is_small_enough_for_the_check_to_mean_
something` guards that property, because it degrades silently as results files are added.

The cost of the narrow haystack is that no documented figure may be a raw per-query score
or a single sweep cell. Neither should be. A claim in prose is about an aggregate, and if
a particular cell matters it belongs in a summary field the script writes out — which is
why `run_sweep.py` records its surface shape and marginals rather than leaving the README
to work them out.

**Settings recorded** requires every file that scores anything to say what produced it,
and **filename agreement** requires that description to match what the filename claims
about the index. These two are a pair: a check that reads a field is worthless against a
file that omits the field. `run_sweep.py` wrote no retriever description, so it swept a
concatenated index through the entire multi-field migration while both checks looked
straight at it.

Numbers that are legitimately not measurements — published baselines quoted from the
paper, thresholds in prose — are in NON_MEASUREMENT with a reason each, and the published
figures are imported from `groundwork.data` rather than repeated, so the allowlist cannot
drift away from what the harness actually checks runs against.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from groundwork.data import REFERENCE_NDCG_10

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"

# (results file, dotted path into the JSON, format spec)
CLAIMS = [
    # Current runs. BM25 indexes title and body as separate fields, as BEIR does.
    ("scifact-bm25.json", "metrics.ndcg@10", ".4f"),
    ("scifact-bm25.json", "metrics.recall@100", ".4f"),
    ("nfcorpus-bm25-graded.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-bm25-graded.json", "metrics.recall@100", ".4f"),
    ("trec-covid-bm25-graded.json", "metrics.ndcg@10", ".4f"),
    ("scidocs-bm25.json", "metrics.ndcg@10", ".4f"),
    ("scidocs-bm25.json", "metrics.recall@100", ".4f"),
    ("scifact-bm25-no-stem.json", "metrics.ndcg@10", ".4f"),
    ("scifact-bm25-no-stopwords.json", "metrics.ndcg@10", ".4f"),
    ("scifact-bm25-plain.json", "metrics.ndcg@10", ".4f"),
    ("scifact-rm3-tuned.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-rm3-tuned.json", "metrics.ndcg@10", ".4f"),
    ("scifact-dense.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-dense.json", "metrics.ndcg@10", ".4f"),
    ("scidocs-dense.json", "metrics.ndcg@10", ".4f"),
    ("scifact-dense-bge.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-dense-bge.json", "metrics.ndcg@10", ".4f"),
    ("scidocs-dense-bge.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-dense-mpnet.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-dense-pubmedbert.json", "metrics.ndcg@10", ".4f"),
    ("scifact-rrf-tuned.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-rrf-tuned.json", "metrics.ndcg@10", ".4f"),
    ("scifact-rrf-bge.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-rrf-bge.json", "metrics.ndcg@10", ".4f"),
    ("scidocs-rrf-k60.json", "metrics.ndcg@10", ".4f"),
    ("scidocs-rrf-bge-k60.json", "metrics.ndcg@10", ".4f"),
    # The three concatenated baselines, which Annexe B quotes to show what the migration
    # moved. They are pinned so that annexe cannot drift into describing runs that are
    # not the ones preserved beside it.
    ("scifact-bm25-singlefield.json", "metrics.ndcg@10", ".4f"),
    ("nfcorpus-bm25-graded-singlefield.json", "metrics.ndcg@10", ".4f"),
    ("trec-covid-bm25-graded-singlefield.json", "metrics.ndcg@10", ".4f"),
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
    """A score without the settings that produced it is not reproducible.

    Sweep files are included, and that is the point. They used to be skipped along with
    comparison files because they have no top-level ``metrics`` block, so a sweep record
    described its tokeniser and nothing else. When the multi-field migration flipped the
    other scripts, `run_sweep.py` was left behind and carried on sweeping a concatenated
    index; because it recorded no retriever, neither this check nor the filename check
    could see it, and the parameters it chose were applied to multi-field runs for a
    while before anyone noticed.
    """
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        scores_something = "metrics" in record or "cells" in record
        if not scores_something:
            continue  # comparison files record a test, not a run
        assert "groundwork_version" in record, f"{path.name} has no version"

        if path.stem.endswith("-singlefield"):
            # Preserved pre-migration runs, some written before the scripts recorded a
            # retriever description at all. They are held to a different requirement
            # rather than an exemption: an archive must say, in the file, that it is one.
            # Back-filling a description would mean writing settings nothing observed.
            assert record.get("note"), (
                f"{path.name} is named as a preserved pre-migration run but carries no "
                "note saying so, which leaves a reader no way to know what it is"
            )
            continue

        assert record.get("retriever") or record.get("retrievers"), (
            f"{path.name} records scores but not what produced them"
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


def _published_reference_figures() -> dict[str, str]:
    """BEIR's published baselines, read from the one place that defines them.

    These are quoted from the papers rather than measured here, so no results file
    contains them. They are imported from ``groundwork.data`` rather than repeated in a
    literal below, so a published figure cannot be quoted in the README without the
    harness also checking runs against it. An allowlist that can drift from what the code
    uses is how this check would get defanged.
    """
    return {
        format(value, ".3f"): f"BEIR's published {name} BM25 figure, from the paper"
        for name, value in REFERENCE_NDCG_10.items()
    }


NON_MEASUREMENT: dict[str, str] = {
    **_published_reference_figures(),
    "0.001": "a threshold in prose ('p < 0.001'), not a measured value",
    "0.0001": "a threshold in prose ('p < 0.0001'), not a measured value",
}


def _variants_in(record: dict) -> set[str]:
    """Every BM25 variant string anywhere in a results file's retriever description."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            variant = node.get("variant")
            if isinstance(variant, str):
                found.add(variant)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(record.get("retriever") or {})
    walk(record.get("retrievers") or {})
    return found


def test_the_filename_agrees_with_how_the_index_was_built():
    """A '-singlefield' file holds concatenated scoring, and everything else does not.

    This exists because the opposite happened. The multi-field migration re-ran some
    results and not others, so for a while several files held concatenated numbers under
    names that now mean multi-field, and an entry in the log quoted them as if they were
    the new ones. Nothing caught it but a reader, late.

    The retriever's own ``describe()`` already records which index built a run, so the
    check is only to insist the filename cannot disagree with it. Comparing a stale file
    against its twin would not do: two runs can legitimately produce identical numbers,
    and by the time the twin exists the damage is already committed.
    """
    for path in sorted(RESULTS.rglob("*.json")):
        variants = _variants_in(json.loads(path.read_text(encoding="utf-8")))
        if not variants & {"lucene", "lucene-multifield"}:
            continue  # no BM25 anywhere in this run
        preserved = path.stem.endswith("-singlefield")
        expected = "lucene" if preserved else "lucene-multifield"
        forbidden = "lucene-multifield" if preserved else "lucene"
        assert expected in variants, (
            f"{path.name} is named as {'single' if preserved else 'multi'}-field but no "
            f"part of it was scored with the {expected} index."
        )
        assert forbidden not in variants, (
            f"{path.name} says {forbidden} in its retriever description, which "
            f"contradicts its filename. Either the run is stale and needs re-running, "
            f"or it is a pre-migration result and its name needs the -singlefield suffix."
        )


# Blocks holding one value per query or per sweep cell. They are excluded from the
# traceable set, and the reason is the whole strength of this check.
#
# The haystack used to be every number in results/, which sounded strict and was not.
# Per-query blocks alone contribute thousands of values, and with that many numbers the
# set of four-decimal strings they cover saturates: 96% of every possible THREE-decimal
# value in [0, 1) appeared somewhere, so a figure invented in prose had a 96% chance of
# matching something by accident and passing. Measured, not assumed - and it had already
# let an uncomputed figure through, the Holm column of the query-dependence table, which
# matched an unrelated fusion delta.
#
# Restricted to summary values, coverage falls to 36% of three-decimal and 5% of
# four-decimal strings. A README figure is quoted to four decimals almost everywhere, so
# a fabricated one now fails about nineteen times in twenty rather than two in five.
#
# The cost is that no README figure may be a raw per-query score or a single sweep cell.
# Neither should be: a claim in the README is about an aggregate, and if a specific cell
# matters it belongs in a summary field that the script writes out.
# "terciles" is deliberately NOT here. It holds three summary means, not one value per
# item, and including it costs 0.2% of coverage while letting the log quote a figure
# the analysis script genuinely computes. The line is per-item bulk, not list-shaped.
BULK_BLOCKS = frozenset({"per_query", "cells", "quantile_grid", "grid"})


def _numeric_values(node: object):
    """Every summary number in a parsed results file, skipping per-item blocks."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in BULK_BLOCKS:
                yield from _numeric_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _numeric_values(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield float(node)


@pytest.fixture(scope="module")
def traceable_figures() -> set[str]:
    """Every summary value in results/, in each string form the README might quote it."""
    figures: set[str] = set()
    for path in sorted(RESULTS.glob("*.json")):
        for value in _numeric_values(json.loads(path.read_text(encoding="utf-8"))):
            magnitude = abs(value)
            for spec in (".2f", ".3f", ".4f"):
                figures.add(format(magnitude, spec))
            figures.add(format(magnitude * 100, ".1f"))
    return figures


def test_the_traceable_set_is_small_enough_for_the_check_to_mean_something():
    """Guard the guard: if the haystack saturates, every check above it is vacuous.

    This is a property of the check itself rather than of any document, and it is a test
    because the failure is silent and was live for several experiments. Adding results
    files grows the traceable set, and at some size "this figure appears in results/"
    stops being evidence that a run produced it. The thresholds are set well above
    current coverage and well below the level at which the check stops discriminating.
    """
    values = {
        value
        for path in sorted(RESULTS.glob("*.json"))
        for value in _numeric_values(json.loads(path.read_text(encoding="utf-8")))
        if 0.0 <= value < 1.0
    }
    three = {format(value, ".3f") for value in values}
    four = {format(value, ".4f") for value in values}
    assert len(three) < 600, (
        f"{len(three)} of 1000 three-decimal values in [0,1) are traceable; the check is "
        "losing its power. Narrow the haystack or pin figures to specific fields."
    )
    assert len(four) < 2000, (
        f"{len(four)} of 10000 four-decimal values in [0,1) are traceable; a fabricated "
        "figure now passes too often for this check to mean anything."
    )


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
