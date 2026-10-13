"""Static figures (matplotlib, Agg). Colours follow a validated categorical order (blue, orange, aqua)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def benchmark(timings: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=130)
    names = ["pandas (baseline)", "Polars LazyFrame"]
    vals = [timings["pandas_seconds"], timings["polars_seconds"]]
    ax.barh(names, vals, color=[ORANGE, BLUE], height=0.5)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.1f} s", va="center", color=INK)
    ax.set_xlabel("Wall time to build all features (s)")
    ax.set_title(
        f"Feature pipeline on {timings['rows']:,} transactions: Polars {timings['speedup']:.1f}x faster",
        color=INK,
        fontsize=10,
    )
    ax.set_xlim(0, max(vals) * 1.2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130)
    for (name, s), col in zip(scores.items(), [BLUE, ORANGE, AQUA]):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=col, linewidth=2, label=name)
    ax.axhline(y.mean(), color=INK2, linestyle=":", linewidth=1, label=f"base rate {y.mean():.2%}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the June 2025 test month", color=INK, fontsize=10)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cost_curve(curves: dict[str, dict], chosen: dict[str, int], path: Path, max_alert_rate: float = 0.008) -> None:
    """Total cost vs alert rate for validation and test; the chosen threshold is marked."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    for (name, cc), col, off in zip(curves.items(), [BLUE, ORANGE], [(30, 75), (70, 30)]):
        n = cc["n"]
        rate = cc["n_alerts"] / n
        keep = rate <= max_alert_rate
        ax.plot(rate[keep] * 100, cc["total_cost"][keep] / 1e3, color=col, linewidth=2, label=name)
        i = chosen[name]
        ax.scatter([rate[i] * 100], [cc["total_cost"][i] / 1e3], s=60, color=col, edgecolor="white", zorder=3)
        ax.annotate(
            f"{name}: threshold {cc['threshold'][i]:.3f}, ${cc['total_cost'][i] / 1e3:,.1f}k",
            (rate[i] * 100, cc["total_cost"][i] / 1e3),
            textcoords="offset points",
            xytext=off,
            fontsize=8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8},
        )
    ax.set_xlabel("Share of transactions alerted / declined (%)")
    ax.set_ylabel("Total expected cost (USD thousands)")
    ax.set_title("Expected cost by threshold (cost matrix in configs/cost_matrix.json)", color=INK, fontsize=10)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> None:
    """Observed fraud rate vs mean predicted probability in log-spaced probability bins (bins with >= 30 rows)."""
    edges = np.concatenate([[0.0], np.logspace(-4, 0, 13)])
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=130)
    for (name, p), col in zip(probs.items(), [ORANGE, BLUE]):
        b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        mp, fy = [], []
        for k in range(len(edges) - 1):
            sel = b == k
            if sel.sum() >= 30:
                mp.append(p[sel].mean())
                fy.append(max(y[sel].mean(), 1e-5))
        ax.plot(mp, fy, marker="o", markersize=5, color=col, linewidth=2, label=name)
    ax.plot([1e-5, 1], [1e-5, 1], color=INK2, linestyle=":", linewidth=1, label="perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability (log-spaced bins)")
    ax.set_ylabel("Observed fraud rate (floored at 1e-5)")
    ax.set_title("Calibration on the test month", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def shap_beeswarm(sv: np.ndarray, x: np.ndarray, features: list[str], path: Path) -> None:
    import shap

    plt.figure()
    shap.summary_plot(sv, x, feature_names=features, show=False, max_display=15, plot_size=(8, 6))
    plt.title("SHAP: global drivers of the fraud score (test sample)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")


def shap_waterfall(sv_row: np.ndarray, base: float, x_row: np.ndarray, features: list[str], title: str, path: Path):
    import shap

    exp = shap.Explanation(values=sv_row, base_values=base, data=x_row, feature_names=features)
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(title, fontsize=9)
    plt.gcf().set_size_inches(8, 5)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")
