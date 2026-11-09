"""Figures for the twin results (all labelled SYNTHETIC)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import CATEGORIES, DCS, STORES  # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED = "#2f6db3", "#e0812f", "#3a9a5b", "#8c8c8c", "#c23b3b"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def validation(val: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(val))
    ax.axvspan(-5, 5, color=GREEN, alpha=0.12, label="±5% acceptance band")
    lo = 100 * (val.twin_p5 / val.historical - 1)
    hi = 100 * (val.twin_p95 / val.historical - 1)
    ax.hlines(y, lo, hi, color=BLUE, lw=3, alpha=0.5, label="twin 5-95% range")
    ax.plot(val.error_pct, y, "o", color=BLUE, label="twin mean")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, val.kpi)
    ax.set_xlabel("twin vs historical (%)")
    ax.set_title("SYNTHETIC: twin reproduces hold-out year KPIs (calibrated on year 1)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def promo_effects(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(df))
    ax.bar(x - 0.25, df.true_lift, 0.25, color=GREY, label="truth (hidden)")
    ax.bar(x, df.posterior_mean, 0.25, color=BLUE, label="PyMC posterior mean")
    ax.errorbar(
        x,
        df.posterior_mean,
        yerr=[df.posterior_mean - df.hdi90_low, df.hdi90_high - df.posterior_mean],
        fmt="none",
        color="k",
        capsize=4,
        label="90% HDI",
    )
    ax.bar(x + 0.25, df.naive_lift, 0.25, color=ORANGE, label="naive promo/no-promo ratio")
    ax.set_xticks(x, df.category)
    ax.set_ylabel("promo demand multiplier")
    ax.set_ylim(1, max(df.naive_lift.max(), df.true_lift.max()) * 1.1)
    ax.set_title("SYNTHETIC: promo lift, confounded naive estimate vs adjusted model")
    ax.legend(fontsize=8)
    _save(fig, path)


def forecast(weeks: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(weeks.week, weeks.p5 / 1e3, weeks.p95 / 1e3, color=BLUE, alpha=0.2, label="90% predictive band")
    ax.plot(weeks.week, weeks.forecast / 1e3, color=BLUE, label="posterior mean (year-1 fit)")
    ax.plot(weeks.week, weeks.actual / 1e3, color="k", lw=1, label="actual units sold")
    ax.set_xlabel("week of hold-out year")
    ax.set_ylabel("units (000s)")
    ax.set_title("SYNTHETIC: weekly network demand, hold-out year (stockout days excluded)")
    ax.legend(fontsize=8)
    _save(fig, path)


def scenarios(dist: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    regimes = list(next(iter(dist.values())).keys())
    fig, axes = plt.subplots(1, len(regimes), figsize=(11, 4.2), sharey=True)
    for ax, reg in zip(np.atleast_1d(axes), regimes):
        names = list(dist)
        data = [dist[n][reg] / 1e6 for n in names]
        ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False)
        for i, d in enumerate(data):
            k = max(1, int(np.ceil(0.05 * len(d))))
            ax.plot(np.sort(d)[:k].mean(), i + 1, "v", color=RED)
        ax.set_yticks(np.arange(1, len(names) + 1), names)
        ax.set_title(f"{reg} regime")
        ax.set_xlabel("operating profit, $M (box 25-75%, whiskers 5-95%, ▼ CVaR5%)")
    fig.suptitle("SYNTHETIC: plan-year profit distribution by strategy")
    _save(fig, path)


def frontier(screen: pd.DataFrame, best: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for east, mk in ((False, "o"), (True, "s")):
        s = screen[screen.open_dc_east == east]
        ax.scatter(
            s.cvar5_profit / 1e6,
            s.expected_profit / 1e6,
            c=np.where(s.feasible, BLUE, GREY),
            marker=mk,
            alpha=0.7,
            label=f"DC-East {'open' if east else 'not open'}",
        )
    ax.scatter(
        [best["cvar5_profit"] / 1e6],
        [best["expected_profit"] / 1e6],
        marker="*",
        s=250,
        color=RED,
        label="chosen (re-evaluated on fresh seeds)",
    )
    ax.set_xlabel("CVaR5% of operating profit, $M")
    ax.set_ylabel("expected operating profit, $M")
    ax.set_title("SYNTHETIC: decision grid screening (blue = feasible)")
    ax.legend(fontsize=8)
    _save(fig, path)


def network(assign_base, assign_east, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, assign, title in ((axes[0], assign_base, "status quo"), (axes[1], assign_east, "with DC-East (MILP)")):
        for s, a in zip(STORES, assign):
            dc = DCS[a]
            ax.plot([s.x, dc.x], [s.y, dc.y], color=GREY, lw=0.8)
        ax.scatter([s.x for s in STORES], [s.y for s in STORES], color=BLUE, label="store")
        used = sorted(set(assign))
        ax.scatter([DCS[d].x for d in used], [DCS[d].y for d in used], color=RED, marker="s", s=80, label="DC")
        for s in STORES:
            ax.annotate(s.name, (s.x, s.y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(title)
        ax.set_xlabel("km")
    axes[0].legend(fontsize=8)
    fig.suptitle("SYNTHETIC network: 12 stores, 2-3 DCs")
    _save(fig, path)


def category_names() -> list[str]:
    return [c.name for c in CATEGORIES]


def one_pager(m: dict, fig_dir: Path, path: Path) -> None:
    """Executive one-page PDF (stretch goal): headline numbers + two figures, all from metrics.json."""
    from matplotlib.backends.backend_pdf import PdfPages

    res = m["scenarios"]["results"]
    base_n, opt_n = res["A_baseline"]["normal"], res["E_optimised"]["normal"]
    opt_s = res["E_optimised"]["stress"]
    d = m["optimiser"]["best_decisions"]
    v = m["validation"]
    lines = [
        "Decision-intelligence digital twin: plan-year recommendation (SYNTHETIC retailer)",
        "",
        f"Recommendation: {'open DC-East' if d['open_dc_east'] else 'keep two DCs'}, nearshore share {d['nearshore_share']:.0%}, "
        f"store safety factor z={d['store_z']}, {d['promo_strategy']} promotions.",
        f"Expected operating profit ${opt_n['expected_profit'] / 1e6:.2f}M vs ${base_n['expected_profit'] / 1e6:.2f}M status quo "
        f"(+${opt_n['delta_profit_vs_baseline_mean'] / 1e6:.2f}M; paired 90% range "
        f"+${opt_n['delta_profit_vs_baseline_p5'] / 1e6:.2f}M to +${opt_n['delta_profit_vs_baseline_p95'] / 1e6:.2f}M).",
        f"Worst-5% years (CVaR): ${opt_n['cvar5_profit'] / 1e6:.2f}M vs ${base_n['cvar5_profit'] / 1e6:.2f}M; "
        f"with a 28-day offshore outage: +${opt_s['delta_profit_vs_baseline_mean'] / 1e6:.2f}M vs status quo.",
        f"Fill rate {opt_n['fill_rate_mean']:.1%} vs {base_n['fill_rate_mean']:.1%}.",
        f"Trust check: calibrated on year 1, the twin reproduced 8 hold-out-year KPIs within {v['max_abs_error_pct']:.1f}% (target 5%).",
        "All data synthetic; figures from results/metrics.json written by `make run`.",
    ]
    with PdfPages(path, metadata={"Title": "Digital twin one-pager (SYNTHETIC)", "CreationDate": None, "ModDate": None}) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.96, lines[0].replace("$", r"\$"), fontsize=13, weight="bold", va="top")
        for i, ln in enumerate(lines[2:]):
            fig.text(0.06, 0.92 - i * 0.025, ln.replace("$", r"\$"), fontsize=8.5, va="top", wrap=True)
        for j, name in enumerate(["scenario_profit.png", "validation_kpis.png"]):
            img = plt.imread(fig_dir / name)
            ax = fig.add_axes([0.05, 0.42 - j * 0.38, 0.9, 0.34])
            ax.imshow(img)
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
