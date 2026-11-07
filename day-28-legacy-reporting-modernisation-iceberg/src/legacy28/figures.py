"""Figures for results/figures (matplotlib, static PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from .rules import RULES, depth  # noqa: E402

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
PALETTE = {"D4": "#2563eb", "D2": "#0d9488", "D3": "#d97706", "D1": "#7c3aed", "D5": "#db2777", "D6": "#64748b"}
ENGINE_COLORS = {"legacy": "#9ca3af", "ibis-duckdb": "#2563eb", "ibis-polars": "#0d9488"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dependency_graph(path: Path) -> None:
    dep = depth()
    layers: dict[int, list[str]] = {}
    for r in RULES:
        layers.setdefault(dep[r.rid], []).append(r.rid)
    pos = {}
    for d, ids in layers.items():
        for i, rid in enumerate(ids):
            pos[rid] = (d * 3.0, -(i - (len(ids) - 1) / 2) * 1.3)
    rm = {r.rid: r for r in RULES}
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for r in RULES:
        for d in r.depends_on:
            (x0, y0), (x1, y1) = pos[d], pos[r.rid]
            ax.annotate(
                "",
                xy=(x1 - 1.15, y1),
                xytext=(x0 + 1.15, y0),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1, "connectionstyle": "arc3,rad=0.12"},
                zorder=1,
            )
    for rid, (x, y) in pos.items():
        r = rm[rid]
        face = "#fde2c8" if r.differences else "#e0e7ff"
        edge = "#c2570c" if r.differences else "#4338ca"
        ax.add_patch(plt.Rectangle((x - 1.15, y - 0.42), 2.3, 0.84, facecolor=face, edgecolor=edge, lw=1.2, zorder=2))
        label = f"{rid}\n{r.name}" + (f"\n[{', '.join(r.differences)}]" if r.differences else "")
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=INK, zorder=3)
    ax.set_xlim(-1.3, max(p[0] for p in pos.values()) + 1.3)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off")
    ax.set_title("Legacy rule dependency graph (orange = rule with a documented intentional difference)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parallel_run(metrics: dict, path: Path) -> None:
    closes = metrics["parallel_run"]["per_close"]
    labels = [c["close"] for c in closes]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    bottom = [0] * len(closes)
    for did, color in PALETTE.items():
        vals = [c["report_cells_changed_by"].get(did, 0) for c in closes]
        ax.bar(labels, vals, bottom=bottom, color=color, label=did, width=0.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    unexplained = [c["unexplained_differences"] for c in closes]
    ax.plot(labels, unexplained, color="#dc2626", marker="o", lw=1.5, label="unexplained")
    _style(ax)
    ax.set_ylabel("report cells changed by each difference\n(close restates all quarters to date)", color=INK, fontsize=9)
    ax.set_title("Parallel run: every difference attributed (SYNTHETIC)", loc="left", fontsize=11, color=INK)
    ax.legend(ncol=7, fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, max(bottom) * 1.25)
    # time travel / engine checks
    tt = metrics["time_travel"]
    checks = {
        "compat == legacy": metrics["parallel_run"]["closes_compat_equals_legacy"],
        "DuckDB == Polars": metrics["parallel_run"]["closes_duckdb_equals_polars"],
        "time travel = filed": tt["filed_quarter_reproduced"],
        "re-run from old inputs": tt["rerun_from_time_travelled_inputs_reproduces_report"],
    }
    ax2.barh(list(checks), list(checks.values()), color="#0d9488", height=0.55)
    for i, v in enumerate(checks.values()):
        ax2.text(v - 0.2, i, f"{v}/{tt['closes']}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, tt["closes"])
    ax2.invert_yaxis()
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=9, colors=INK)
    ax2.set_title("Quarterly closes passing each check", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def benchmark(metrics: dict, path: Path) -> None:
    rows = metrics["benchmark"]
    scales = sorted({r["scale"] for r in rows})
    engines = ["legacy", "ibis-duckdb", "ibis-polars"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.26
    for i, e in enumerate(engines):
        vals = [next(r["wall_seconds"] for r in rows if r["scale"] == s and r["engine"] == e) for s in scales]
        xs = [j + (i - 1) * w for j in range(len(scales))]
        ax.bar(xs, vals, width=w, color=ENGINE_COLORS[e], label=e)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.08, f"{v:.1f}s", ha="center", fontsize=8, color=INK)
    pol = {r["scale"]: r["policies"] for r in rows}
    ax.set_xticks(range(len(scales)), [f"{pol[s]:,} policies" for s in scales])
    _style(ax)
    ax.set_ylabel("wall seconds, full 8-quarter run", fontsize=9, color=INK)
    ax.set_title("Runtime: legacy row-by-row vs the same Ibis code on two engines", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _money(v: float) -> str:
    sign = "-" if v < 0 else "+"
    return f"{sign}${abs(v):,.2f}" if abs(v) < 10 else f"{sign}${abs(v):,.0f}"


def _axis_money(x: float, lim: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x) / 1e6:,.1f}M" if lim >= 1e6 else f"{sign}${abs(x) / 1e3:,.0f}k"


def impact(metrics: dict, path: Path) -> None:
    imp = metrics["impact_by_difference_final_close"]
    ids = list(imp)
    measures = [("other_lob_earned_cents", "earned premium reported as OTHER"), ("incurred_cents", "incurred claims"), ("ibnr_cents", "IBNR")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (m, name) in zip(axes, measures):
        vals = [imp[d][m] / 100 for d in ids]
        ax.barh(ids, vals, color=[PALETTE[d] for d in ids])
        ax.axvline(0, color=MUTED, lw=0.8)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {_money(v)} ", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
        ax.set_title(f"change in {name}", fontsize=10, color=INK, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        lim = max(abs(v) for v in vals) or 1
        ax.set_xlim(-lim * 1.7, lim * 1.7)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _, lim=lim: _axis_money(x, lim)))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axes[0].invert_yaxis()
    fig.suptitle("Effect of each intentional difference, 2025Q4 close, all 8 quarters (SYNTHETIC)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(metrics: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dependency_graph(out / "dependency_graph.png")
    parallel_run(metrics, out / "parallel_run.png")
    benchmark(metrics, out / "benchmark.png")
    impact(metrics, out / "impact_by_difference.png")
