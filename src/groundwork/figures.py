"""Static SVG figures for the README, drawn from committed results.

Written by hand rather than with a plotting library, because the core install is numpy
and tqdm and a README chart is not a reason to add a third dependency. The output is
deterministic, so a test can regenerate a committed figure and fail if it has drifted
from the results it claims to show.
"""

import math
from dataclasses import dataclass
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class Row:
    """One method: a label, the colour group it belongs to, and one value per panel."""

    label: str
    group: str
    values: tuple[float, ...]
    annotate: bool = False  # print the value beside each dot


@dataclass(frozen=True)
class Group:
    """A colour and a legend entry, with separate light and dark steps."""

    name: str
    light: str
    dark: str


# Surfaces and ink for each theme. The dark values are their own steps rather than an
# inversion, so a README viewed in a dark theme stays readable.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#383835"},
}

# Layout, in SVG user units.
WIDTH = 900
LABEL_WIDTH = 270
PANEL_GAP = 52
MARGIN = 24
HEADER = 128
ROW_HEIGHT = 26
AXIS_HEIGHT = 44
DOT_RADIUS = 5


def nice_ticks(low: float, high: float, max_ticks: int = 7) -> list[float]:
    """Evenly spaced round values covering [low, high], at most ``max_ticks`` of them.

    The step is the smallest of 1, 2, 2.5 or 5 times a power of ten that keeps the tick
    count within the limit; the first tick is at or below ``low`` and the last at or
    above ``high``.
    """
    if high <= low:
        raise ValueError(f"empty range: {low} to {high}")
    magnitude = 10 ** math.floor(math.log10(high - low))
    for base in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10):
        step = base * magnitude
        first = math.floor(low / step + 1e-9) * step
        last = math.ceil(high / step - 1e-9) * step
        count = round((last - first) / step) + 1
        if count <= max_ticks:
            return [round(first + i * step, 10) for i in range(count)]
    raise ValueError(f"no step found for {low} to {high}")  # unreachable for base 10


def scale(value: float, domain: tuple[float, float], extent: tuple[float, float]) -> float:
    """Map ``value`` linearly from ``domain`` onto ``extent``."""
    (d0, d1), (e0, e1) = domain, extent
    return e0 + (value - d0) / (d1 - d0) * (e1 - e0)


def _decimals(ticks: list[float]) -> int:
    step = ticks[1] - ticks[0]
    return max(0, -math.floor(math.log10(step) + 1e-9))


def dot_plot_svg(
    rows: list[Row],
    panels: list[str],
    groups: list[Group],
    reference: Row,
    title: str,
    subtitle: str,
    caption: str,
) -> str:
    """A dot plot: one row per method, one panel per dataset, with a reference line.

    Dots rather than bars because the axes do not start at zero. Each panel gets its own
    range because the datasets score on different scales, and a shared axis would flatten
    the smaller one into a line.
    """
    for row in [*rows, reference]:
        if len(row.values) != len(panels):
            raise ValueError(f"{row.label!r} has {len(row.values)} values for {len(panels)} panels")
    known = {g.name for g in groups}
    unknown = {row.group for row in rows} - known
    if unknown:
        raise ValueError(f"rows use groups with no colour: {sorted(unknown)}")

    all_rows = [*rows, reference]
    plot_top = HEADER
    plot_height = ROW_HEIGHT * len(all_rows)
    height = plot_top + plot_height + AXIS_HEIGHT + 28
    panel_width = (WIDTH - MARGIN - LABEL_WIDTH - PANEL_GAP * (len(panels) - 1)) / len(panels)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'width="{WIDTH}" height="{height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{escape(title)}</title>',
        f'<desc id="d">{escape(subtitle)}</desc>',
        _style(groups),
        f'<rect class="surface" width="{WIDTH}" height="{height}" rx="8"/>',
        f'<text class="title" x="{MARGIN}" y="34">{escape(title)}</text>',
        f'<text class="ink2" x="{MARGIN}" y="56">{escape(subtitle)}</text>',
    ]
    out += _legend(groups, reference.label, y=84)

    for i, row in enumerate(all_rows):
        y = plot_top + ROW_HEIGHT * (i + 0.5)
        weight = " strong" if row.annotate else ""
        out.append(
            f'<text class="ink{weight}" x="{MARGIN}" y="{y + 4:.1f}">{escape(row.label)}</text>'
        )

    for p, name in enumerate(panels):
        x0 = LABEL_WIDTH + p * (panel_width + PANEL_GAP)
        x1 = x0 + panel_width
        values = [row.values[p] for row in all_rows]
        ticks = nice_ticks(min(values), max(values))
        domain = (ticks[0], ticks[-1])
        digits = _decimals(ticks)

        out.append(f'<text class="panel" x="{x0}" y="{plot_top - 10}">{escape(name)}</text>')
        for tick in ticks:
            x = scale(tick, domain, (x0, x1))
            out.append(
                f'<line class="grid" x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                f'y2="{plot_top + plot_height}"/>'
            )
            out.append(
                f'<text class="ink2 tick" x="{x:.1f}" y="{plot_top + plot_height + 18}">'
                f"{tick:.{digits}f}</text>"
            )

        ref_x = scale(reference.values[p], domain, (x0, x1))
        out.append(
            f'<line class="reference" x1="{ref_x:.1f}" y1="{plot_top}" x2="{ref_x:.1f}" '
            f'y2="{plot_top + plot_height}"/>'
        )

        for i, row in enumerate(all_rows):
            y = plot_top + ROW_HEIGHT * (i + 0.5)
            x = scale(row.values[p], domain, (x0, x1))
            css = "reference-dot" if row is reference else f"g{_index(groups, row.group)}"
            out.append(
                f'<circle class="{css}" cx="{x:.1f}" cy="{y:.1f}" r="{DOT_RADIUS}">'
                f"<title>{escape(row.label)}, {escape(name)}: {row.values[p]:.4f}</title></circle>"
            )
            if row.annotate:
                out.append(
                    f'<text class="ink strong value" x="{x + DOT_RADIUS + 5:.1f}" '
                    f'y="{y + 4:.1f}">{row.values[p]:.4f}</text>'
                )

    out.append(f'<text class="ink2 small" x="{MARGIN}" y="{height - 14}">{escape(caption)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _index(groups: list[Group], name: str) -> int:
    return next(i for i, g in enumerate(groups) if g.name == name)


def _legend(groups: list[Group], reference_label: str, y: int) -> list[str]:
    parts, x = [], MARGIN
    for i, group in enumerate(groups):
        parts.append(f'<circle class="g{i}" cx="{x + DOT_RADIUS}" cy="{y - 4}" r="{DOT_RADIUS}"/>')
        parts.append(f'<text class="ink" x="{x + 16}" y="{y}">{escape(group.name)}</text>')
        x += 16 + 7.2 * len(group.name) + 28
    parts.append(f'<line class="reference" x1="{x}" y1="{y - 12}" x2="{x}" y2="{y + 4}"/>')
    parts.append(f'<text class="ink" x="{x + 10}" y="{y}">{escape(reference_label)}</text>')
    return parts


def _style(groups: list[Group]) -> str:
    def rules(theme: str) -> str:
        t = THEMES[theme]
        colours = "".join(f".g{i}{{fill:{getattr(g, theme)}}}" for i, g in enumerate(groups))
        return (
            f".surface{{fill:{t['surface']}}}"
            f".title,.ink,.panel{{fill:{t['ink']}}}.ink2{{fill:{t['ink2']}}}"
            f".grid{{stroke:{t['grid']}}}.reference{{stroke:{t['ink2']}}}"
            f".reference-dot{{fill:{t['ink2']}}}"
            f"circle{{stroke:{t['surface']}}}{colours}"
        )

    return (
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,"
        "sans-serif;font-size:13px}"
        ".title{font-size:18px;font-weight:600}.panel{font-size:14px;font-weight:600}"
        ".strong{font-weight:600}.small{font-size:11.5px}.tick{text-anchor:middle;font-size:12px}"
        ".value{font-size:12px}"
        ".grid{stroke-width:1}.reference{stroke-width:1.5;stroke-dasharray:4 3}"
        "circle{stroke-width:2}"
        f"{rules('light')}"
        f"@media (prefers-color-scheme:dark){{{rules('dark')}}}"
        "</style>"
    )
