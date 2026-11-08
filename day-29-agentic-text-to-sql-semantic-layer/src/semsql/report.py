"""Figures and Markdown reports written to results/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .benchmark import Outcome  # noqa: E402

BLUE, ORANGE, TEXT, TEXT2, SURFACE = (
    "#2a78d6",
    "#eb6834",
    "#0b0b0b",
    "#52514e",
    "#fcfcfb",
)
CAT_ORDER = [
    "simple",
    "by_dimension",
    "time_series",
    "filtered",
    "point_in_time",
    "ranking",
    "multi_metric",
    "refuse_write",
    "refuse_pii",
    "refuse_undefined",
    "clarify",
]


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TEXT2)
    ax.spines["bottom"].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def bar_compare(
    series: dict[str, dict[str, dict[str, Any]]],
    order: list[str],
    title: str,
    path: Path,
) -> None:
    names = list(series)
    cats = [c for c in order if c in series[names[0]]]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(cats) + 1.6), facecolor=SURFACE)
    h = 0.38
    colors = [BLUE, ORANGE]
    for k, name in enumerate(names):
        vals = [series[name][c]["accuracy"] * 100 for c in cats]
        ys = [i + (k - 0.5) * h for i in range(len(cats))]
        ax.barh(ys, vals, height=h, color=colors[k], label=name)
        for y, v, c in zip(ys, vals, cats):
            d = series[name][c]
            ax.text(
                v + 1,
                y,
                f"{d['correct']}/{d['n']}",
                va="center",
                fontsize=8,
                color=TEXT2,
            )
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("accuracy (%)", color=TEXT2)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=26)
    _style(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def failure_md(name: str, outs: list[Outcome]) -> str:
    fails = [o for o in outs if not o.correct]
    lines = [f"### {name}: {len(fails)} failures of {len(outs)}", ""]
    if not fails:
        return "\n".join(lines + ["None.", ""])
    lines += [
        "| id | category | question | outcome | reason | traps |",
        "|---|---|---|---|---|---|",
    ]
    for o in fails:
        lines.append(f"| {o.id} | {o.category} | {o.question} | {o.status} | {o.reason} | {', '.join(o.traps)} |")
    return "\n".join(lines + [""])
