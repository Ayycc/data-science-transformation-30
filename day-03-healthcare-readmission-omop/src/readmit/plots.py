"""Matplotlib figures (static PNG). Series colours follow a fixed categorical order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

COLORS = {"logistic_l2": "#2a78d6", "gbm_calibrated": "#eb6834", "lace": "#1baf7a"}
LABELS = {"logistic_l2": "L2 logistic regression", "gbm_calibrated": "Gradient boosting (calibrated)", "lace": "LACE"}
NEUTRAL = "#8a8985"
INK = "#2b2b29"
SUBTLE = "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e7e6e2",
        "grid.linewidth": 0.8,
        "xtick.color": SUBTLE,
        "ytick.color": SUBTLE,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def calibration_plot(bins: dict[str, pd.DataFrame], preds: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 6.6), height_ratios=[3, 1], sharex=True)
    hi = max(float(b["mean_predicted"].max()) for b in bins.values())
    hi = max(hi, max(float(b["ci_high"].max()) for b in bins.values())) * 1.05
    ax.plot([0, hi], [0, hi], color=NEUTRAL, lw=1, ls="--", label="Perfect calibration")
    for name, b in bins.items():
        ax.errorbar(
            b["mean_predicted"],
            b["observed"],
            yerr=[(b["observed"] - b["ci_low"]).clip(lower=0), (b["ci_high"] - b["observed"]).clip(lower=0)],
            color=COLORS[name],
            marker="o",
            ms=5,
            lw=2,
            capsize=0,
            elinewidth=1,
            label=LABELS[name],
        )
    ax.set_ylabel("Observed 30-day readmission rate")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    edges = np.linspace(0, hi, 40)
    for name, p in preds.items():
        axh.hist(p, bins=edges, histtype="step", color=COLORS[name], lw=1.5)
    axh.set_xlabel("Predicted risk (decile bins, 95% Wilson intervals)")
    axh.set_ylabel("Admissions")
    _save(fig, path)


def decision_curve_plot(dca: pd.DataFrame, models: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(dca["threshold"], dca["treat_all"], color=NEUTRAL, lw=1.2, label="Treat all")
    ax.axhline(0, color=INK, lw=0.8, label="Treat none")
    for m in models:
        ax.plot(dca["threshold"], dca[m], color=COLORS[m], label=LABELS[m])
    top = float(dca[models + ["treat_all"]].max().max())
    ax.set_ylim(-0.02, top * 1.15)
    ax.set_xlabel("Threshold probability for intervention")
    ax.set_ylabel("Net benefit")
    ax.set_title(title, loc="left")
    ax.legend()
    _save(fig, path)


def roc_plot(y: np.ndarray, preds: dict[str, np.ndarray], aucs: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[name], label=f"{LABELS[name]} (AUROC {aucs[name]:.3f})")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC on the 2024 test period", loc="left")
    ax.legend(loc="lower right")
    _save(fig, path)


def subgroup_plot(sub: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=False)
    sub = sub[sub["n"] >= 30].reset_index(drop=True)
    labels = [f"{r.level}\n(n={r.n:,})" for r in sub.itertuples()]
    x = np.arange(len(sub))
    sep = int((sub["dimension"] == "age_band").sum())
    for ax, col, ref, lab in (
        (axes[0], "auroc", None, "AUROC"),
        (axes[1], "o_e_ratio", 1.0, "Observed / expected"),
    ):
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.bar(x, sub[col], color="#2a78d6", width=0.62)
        if ref is not None:
            ax.axhline(ref, color=INK, lw=0.9)
        ax.axvline(sep - 0.5, color="#c9c8c2", lw=1)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel(lab)
        for xi, v in zip(x, sub[col]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0.0 if col == "o_e_ratio" else 0.5, max(1.0, float(sub[col].max())) * 1.12)
    axes[0].set_title(title, loc="left")
    _save(fig, path)


def km_plot(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    shades = ["#b7d3f2", "#7aaee8", "#2a78d6", "#0d3f80"]
    for (name, c), col in zip(curves.items(), shades):
        ax.step(c["day"], c["cumulative_incidence"], where="post", color=col, label=name)
        ax.text(
            30.4,
            float(c["cumulative_incidence"].iloc[-1]),
            f"{c['cumulative_incidence'].iloc[-1]:.1%}",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xlim(0, 33)
    ax.set_xlabel("Days since discharge")
    ax.set_ylabel("Cumulative unplanned readmission (1 - KM)")
    ax.set_title("Time to readmission by predicted-risk quartile (test)", loc="left")
    ax.legend(title="Risk quartile", loc="upper left")
    _save(fig, path)
