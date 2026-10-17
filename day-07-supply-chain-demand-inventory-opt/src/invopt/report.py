"""Figures for results/ (matplotlib, PNG). All data shown are SYNTHETIC."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {
    "heuristic": "Current heuristic",
    "optimised_greedy": "Optimised (s,S), greedy allocation",
    "optimised_milp": "Optimised (s,S) + MILP",
}
COLORS = {"heuristic": ORANGE, "optimised_greedy": AQUA, "optimised_milp": BLUE}


def _style(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(results: Path, prep: dict, runs: dict, inputs: dict, summ: dict, sens: list, trace_items) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelcolor": INK})

    # 1) Cost breakdown per policy (stacked horizontal bars, $k).
    comps = [
        ("holding_cost", "Holding"),
        ("shortage_cost", "Lost sales"),
        ("ordering_cost", "Ordering"),
        ("seasonal_overage_cost", "Seasonal leftovers"),
    ]
    shades = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    names = list(summ)
    for j, name in enumerate(names):
        left = 0.0
        for (key, lab), col in zip(comps, shades):
            val = summ[name][key] / 1e3
            ax.barh(j, val, left=left, color=col, edgecolor="white", linewidth=2, label=lab if j == 0 else None, height=0.6)
            left += val
        ax.text(left + 5, j, f"${left:,.0f}k  | fill {summ[name]['fill_rate']:.1%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Simulated annual cost ($ thousands, SYNTHETIC test year)")
    ax.set_xlim(0, max(summ[n]["total_cost"] for n in names) / 1e3 * 1.45)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=9)
    ax.set_title("Total cost by policy (SimPy, 52 weeks)", loc="left")
    _style(ax)
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    fig.savefig(results / "cost_breakdown.png", dpi=130)
    plt.close(fig)

    # 2) Fill rate by ABC class.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cls = ["A", "B", "C", "seasonal"]
    w = 0.26
    for j, name in enumerate(names):
        vals = [summ[name][f"fill_rate_{c}"] or 0 for c in cls]
        ax.bar(np.arange(len(cls)) + (j - 1) * w, vals, width=w - 0.02, color=COLORS[name], label=LABELS[name])
    ax.set_xticks(range(len(cls)), ["A items", "B items", "C items", "Seasonal (newsvendor)"])
    lo = min(min(summ[n][f"fill_rate_{c}"] or 1 for c in cls) for n in names)
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Unit fill rate")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Fill rate by segment (SYNTHETIC)", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(results / "fill_rate_by_class.png", dpi=130)
    plt.close(fig)

    # 3) Weekly inventory value.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for name in names:
        ax.plot(runs[name].weekly_inventory_value / 1e3, color=COLORS[name], linewidth=2, label=LABELS[name])
    ax.set_xlabel("Test-year week")
    ax.set_ylabel("On-hand value ($k)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Inventory value through the simulated year", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(results / "inventory_value_weekly.png", dpi=130)
    plt.close(fig)

    # 4) ABC x XYZ heatmap (counts).
    seg = prep["seg"]
    tab = (
        seg.groupby(["abc", "xyz"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(tab.to_numpy(), cmap="Blues")
    for (r, c), v in np.ndenumerate(tab.to_numpy()):
        ax.text(c, r, str(v), ha="center", va="center", color="white" if v > tab.to_numpy().max() * 0.55 else INK)
    ax.set_xticks(range(3), ["X (CV<0.5)", "Y", "Z (CV>=1)"])
    ax.set_yticks(range(3), ["A (80% rev)", "B", "C"])
    ax.set_title("Items per ABC x XYZ segment", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(results / "abc_xyz_heatmap.png", dpi=130)
    plt.close(fig)

    # 5) Sensitivity to lead-time variability.
    if sens:
        df = pd.DataFrame(sens)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for name in ("heuristic", "optimised_milp"):
            d = df[df["policy"] == name]
            axes[0].plot(
                d["lt_cv_multiplier"],
                d["total_cost"] / 1e3,
                "o-",
                color=COLORS[name],
                linewidth=2,
                markersize=7,
                label=LABELS[name],
            )
            axes[1].plot(d["lt_cv_multiplier"], d["fill_rate"], "o-", color=COLORS[name], linewidth=2, markersize=7)
        axes[0].set_ylabel("Total cost ($k)")
        axes[1].set_ylabel("Unit fill rate")
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for a in axes:
            a.set_xlabel("Lead-time CV multiplier")
            _style(a)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Sensitivity to supplier lead-time variability (SYNTHETIC)", x=0.02, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(results / "sensitivity_lead_time_cv.png", dpi=130)
        plt.close(fig)

    # 6) Example on-hand trajectories.
    if runs["optimised_milp"].on_hand_trace is not None:
        fig, axes = plt.subplots(1, len(trace_items), figsize=(10, 2.8), sharex=True)
        for a, j in zip(np.atleast_1d(axes), range(len(trace_items))):
            for name in ("heuristic", "optimised_milp"):
                a.step(
                    range(prep["T"]),
                    runs[name].on_hand_trace[j],
                    where="post",
                    color=COLORS[name],
                    linewidth=1.8,
                    label=LABELS[name],
                )
            item = seg.index[trace_items[j]]
            a.set_title(item.replace("|", " @ "), fontsize=9, loc="left")
            a.set_xlabel("Week")
            _style(a)
        np.atleast_1d(axes)[0].set_ylabel("On hand (units)")
        np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(results / "example_trajectories.png", dpi=130)
        plt.close(fig)
