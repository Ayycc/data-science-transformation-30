"""Static figures (matplotlib, light surface, fixed categorical order)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def ps_overlap(e: np.ndarray, t: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(e[t == 1], bins=bins, alpha=0.75, color=S1, label="enrolled", edgecolor=SURFACE, linewidth=1)
    ax.hist(e[t == 0], bins=bins, alpha=0.75, color=S2, label="not enrolled", edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("estimated propensity score")
    ax.set_ylabel("people")
    ax.set_title("Overlap of estimated propensity scores (SYNTHETIC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def love_plot(bal: pd.DataFrame, path: Path) -> None:
    b = bal.sort_values("smd_raw", key=np.abs)
    y = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvline(0, color=INK2, lw=0.8)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=INK2, lw=0.8, ls=":")
    ax.scatter(b["smd_raw"], y, color=S2, s=40, label="unadjusted", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(b["smd_ipw"], y, color=S1, s=40, label="IPW", marker="s", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(b["smd_matched"], y, color=S3, s=40, label="matched", marker="^", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, b.index)
    ax.set_xlabel("standardised mean difference (enrolled - not enrolled); dotted = +/-0.1")
    ax.set_title("Covariate balance before and after propensity adjustment (SYNTHETIC)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def estimator_forest(est: pd.DataFrame, truths: dict[str, float], path: Path) -> None:
    outcomes = list(truths)
    fig, axes = plt.subplots(1, len(outcomes), figsize=(12, 4.6), sharey=True)
    for ax, yname in zip(np.atleast_1d(axes), outcomes, strict=True):
        d = est[est["outcome"] == yname].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        ax.axvline(truths[yname], color=INK, ls="--", lw=1.2, label="true SATE")
        ax.axvline(0, color=INK2, lw=0.6)
        for i, r in d.iterrows():
            col = S2 if r["method"].startswith("Naive") or "certificate" in r["method"] else S1
            if pd.notna(r["ci_low"]):
                ax.plot([r["ci_low"], r["ci_high"]], [y[i], y[i]], color=col, lw=2, solid_capstyle="round")
            ax.scatter([r["estimate"]], [y[i]], color=col, s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.set_yticks(y, d["method"])
        ax.set_title(f"{yname}: estimate and 95% CI\n(orange = biased by design)")
        ax.legend(loc="lower left")
    fig.suptitle("Naive vs adjusted estimates against the known truth (SYNTHETIC)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cate_figure(cate: np.ndarray, true_tau: np.ndarray, sub: pd.DataFrame, path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(cate), size=min(3000, len(cate)), replace=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.scatter(true_tau[ix], cate[ix], s=8, alpha=0.35, color=S1, edgecolor="none")
    lim = [min(true_tau.min(), cate.min()), max(true_tau.max(), cate.max())]
    a.plot(lim, lim, color=INK, ls="--", lw=1)
    a.set_xlabel("true individual effect on earnings ($)")
    a.set_ylabel("causal-forest CATE ($)")
    a.set_title(f"CATE vs truth (3,000-person sample), corr = {np.corrcoef(cate, true_tau)[0, 1]:.2f}")
    s = sub[sub["group"] != "all"].reset_index(drop=True)
    y = np.arange(len(s))[::-1]
    b.plot([s["ci_low"], s["ci_high"]], [y, y], color=S1, lw=2, solid_capstyle="round")
    b.scatter(s["cate_mean"], y, color=S1, s=36, zorder=3, label="estimated (95% CI)", edgecolor=SURFACE, linewidth=1.5)
    b.scatter(s["true_effect"], y, color=INK, marker="|", s=160, zorder=4, label="truth")
    b.set_yticks(y, s["group"])
    b.set_xlabel("average effect on annual earnings ($)")
    b.set_title("Who benefits most: subgroup effects")
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def event_study(cs: pd.DataFrame, truth: dict[int, float], tw: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(-0.5, color=INK2, lw=0.8, ls=":")
    ax.fill_between(cs["event_time"], cs["ci_low"], cs["ci_high"], color=S1, alpha=0.18, linewidth=0)
    ax.plot(cs["event_time"], cs["att"], color=S1, lw=2, marker="o", ms=4, label="Callaway-Sant'Anna (95% bootstrap CI)")
    ax.plot(tw["event_time"], tw["coef"], color=S2, lw=2, marker="s", ms=4, label="dynamic TWFE (binned ends)")
    te = sorted(truth)
    ax.plot([-1] + te, [0] + [truth[e] for e in te], color=INK, ls="--", lw=1.2, label="true effect")
    ax.set_xlabel("quarters since programme start (base = -1)")
    ax.set_ylabel("employment rate effect (pp)")
    ax.set_title("Staggered rollout event study, 60 SYNTHETIC counties")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def synth_figure(series: pd.DataFrame, t0: int, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(series["quarter"], series["observed"], color=S1, lw=2, label="pilot county")
    a.plot(series["quarter"], series["synthetic"], color=S2, lw=2, ls="-", label="synthetic control")
    a.axvline(t0 - 0.5, color=INK2, ls=":")
    a.set_xlabel("quarter")
    a.set_ylabel("employment rate (%)")
    a.set_title("Pilot county vs its synthetic control (SYNTHETIC)")
    a.legend()
    b.axhline(0, color=INK2, lw=0.8)
    b.plot(series["quarter"], series["gap"], color=S1, lw=2, label="estimated gap")
    b.plot(series["quarter"], series["true_effect"], color=INK, ls="--", lw=1.2, label="true effect")
    b.axvline(t0 - 0.5, color=INK2, ls=":")
    b.set_xlabel("quarter")
    b.set_ylabel("pp")
    b.set_title("Gap = observed - synthetic")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
