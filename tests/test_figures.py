import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from groundwork.figures import Group, Row, dot_plot_svg, nice_ticks, scale

ROOT = Path(__file__).resolve().parent.parent
SVG = "{http://www.w3.org/2000/svg}"


# --- nice_ticks --------------------------------------------------------------------------


def test_ticks_for_the_scifact_range():
    # Range 0.6550 to 0.7207 spans 0.0657, so candidate steps are multiples of 0.01.
    # Step 0.01: 0.65 .. 0.73 is 9 ticks, over the limit of 7.
    # Step 0.02: floor(0.6550 / 0.02) = 32 -> 0.64; ceil(0.7207 / 0.02) = 37 -> 0.74.
    # That is 0.64, 0.66, 0.68, 0.70, 0.72, 0.74: 6 ticks.
    assert nice_ticks(0.6550, 0.7207) == pytest.approx([0.64, 0.66, 0.68, 0.70, 0.72, 0.74])


def test_ticks_for_the_nfcorpus_range():
    # Range 0.3155 to 0.3610. Step 0.005: 0.315 .. 0.365 is 11 ticks, over the limit.
    # Step 0.01: floor(31.55) = 31 -> 0.31; ceil(36.10) = 37 -> 0.37: 7 ticks, allowed.
    assert nice_ticks(0.3155, 0.3610) == pytest.approx([0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37])


def test_ticks_on_the_unit_interval():
    # Step 0.1 gives 11 ticks; step 0.2 gives 0, 0.2, 0.4, 0.6, 0.8, 1.0.
    assert nice_ticks(0.0, 1.0) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_ticks_cover_a_range_that_sits_on_a_tick():
    # 0.30 and 0.36 are both multiples of 0.01, so no padding tick is added either side.
    assert nice_ticks(0.30, 0.36) == pytest.approx([0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36])


def test_ticks_reject_an_empty_range():
    with pytest.raises(ValueError, match="empty range"):
        nice_ticks(0.5, 0.5)


# --- scale -------------------------------------------------------------------------------


def test_scale_maps_endpoints_and_midpoint():
    assert scale(0.64, (0.64, 0.74), (100.0, 300.0)) == pytest.approx(100.0)
    assert scale(0.74, (0.64, 0.74), (100.0, 300.0)) == pytest.approx(300.0)
    assert scale(0.69, (0.64, 0.74), (100.0, 300.0)) == pytest.approx(200.0)


# --- dot_plot_svg ---------------------------------------------------------------------------

GROUPS = [Group("A", light="#2a78d6", dark="#3987e5"), Group("B", light="#8a8984", dark="#8a8984")]
REFERENCE = Row("Baseline", "B", (0.50, 0.20))


def _plot(rows: list[Row]) -> ET.Element:
    svg = dot_plot_svg(rows, ["P1", "P2"], GROUPS, REFERENCE, title="T", subtitle="S", caption="C")
    return ET.fromstring(svg)


def test_one_dot_per_method_per_panel():
    rows = [Row("m1", "A", (0.60, 0.25)), Row("m2", "B", (0.55, 0.22))]
    root = _plot(rows)
    data_dots = [c for c in root.iter(f"{SVG}circle") if c.find(f"{SVG}title") is not None]
    # Two methods plus the reference row, in each of two panels.
    assert len(data_dots) == 6
    labels = sorted(c.find(f"{SVG}title").text for c in data_dots)
    assert "m1, P1: 0.6000" in labels
    assert "Baseline, P2: 0.2000" in labels


def test_only_annotated_rows_print_their_values():
    rows = [Row("m1", "A", (0.60, 0.25), annotate=True), Row("m2", "B", (0.55, 0.22))]
    texts = [t.text for t in _plot(rows).iter(f"{SVG}text")]
    assert "0.6000" in texts
    assert "0.2500" in texts
    assert "0.5500" not in texts


def test_a_row_with_the_wrong_number_of_values_is_rejected():
    with pytest.raises(ValueError, match="has 1 values for 2 panels"):
        _plot([Row("m1", "A", (0.60,))])


def test_a_row_in_an_unknown_group_is_rejected():
    with pytest.raises(ValueError, match="no colour"):
        _plot([Row("m1", "Z", (0.60, 0.25))])


# --- the committed figure ---------------------------------------------------------------


def test_the_committed_readme_figure_matches_the_results():
    """The chart at the top of the README is drawn from results/, and must stay so.

    Regenerating it here means a re-run that moves a number, or a hand edit to the SVG,
    fails the build rather than leaving a picture that disagrees with the tables below it.
    """
    spec = importlib.util.spec_from_file_location(
        "plot_methods", ROOT / "scripts" / "plot_methods.py"
    )
    plot_methods = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_methods)
    committed = (ROOT / "docs" / "figures" / "ndcg10-by-method.svg").read_text(encoding="utf-8")
    assert plot_methods.build() == committed, (
        "docs/figures/ndcg10-by-method.svg is stale; run python scripts/plot_methods.py"
    )
