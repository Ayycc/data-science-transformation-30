"""Figures (matplotlib, Agg). All data shown is SYNTHETIC."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import CHANNELS, HDI_PROB  # noqa: E402

COLORS = {"tv": "#4C72B0", "search": "#DD8452", "social": "#55A868", "retail_media": "#8172B3"}
ALLOC_COLORS = {"current": "#4C72B0", "expected_optimal": "#DD8452", "risk_adjusted": "#55A868", "last_click_guided": "#C44E52"}
Q = ((1 - HDI_PROB) / 2, 1 - (1 - HDI_PROB) / 2)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def predictive_bands(dates, observed_nat: np.ndarray, prior_nat: np.ndarray, post_nat: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, draws, title, col in ((axes[0], prior_nat, "Prior predictive", "#999999"), (axes[1], post_nat, "Posterior predictive", "#4C72B0")):
        lo, hi = np.quantile(draws, Q, axis=0)
        ax.fill_between(dates, lo / 1e3, hi / 1e3, color=col, alpha=0.35, label="94% interval")
        ax.plot(dates, np.median(draws, axis=0) / 1e3, color=col, lw=1.2, label="median")
        ax.plot(dates, observed_nat / 1e3, color="black", lw=0.9, label="observed")
        ax.set_title(f"{title}: national weekly sales ($M, SYNTHETIC)")
        ax.set_ylabel("$M / week")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    _save(fig, path)


def roas_forest(roas: pd.DataFrame, roas_draws: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in roas.reset_index(drop=True).iterrows():
        c = row["channel"]
        parts = ax.violinplot(roas_draws[:, i], positions=[i], vert=False, showextrema=False, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(COLORS[c])
            b.set_alpha(0.45)
        ax.plot([row["roas_hdi_low"], row["roas_hdi_high"]], [i, i], color=COLORS[c], lw=3)
        ax.scatter(row["true_roas"], i, marker="D", color="black", zorder=5, label="true ROAS" if i == 0 else None)
        ax.scatter(row["last_click_roas"], i, marker="x", color="red", s=60, zorder=5, label="last-click ROAS" if i == 0 else None)
    ax.set_yticks(range(len(roas)), roas["channel"])
    ax.set_xlabel("ROAS ($ incremental sales per $ spend)")
    ax.set_title("ROAS posterior (94% HDI bar) vs truth and last-click (SYNTHETIC)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def recovery_plot(rec: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, grp, title in ((axes[0], "alpha", "Adstock decay alpha"), (axes[1], "lam", "Saturation lam")):
        d = rec[rec["group"] == grp].reset_index(drop=True)
        ax.errorbar(
            d["mean"], range(len(d)), xerr=[d["mean"] - d["hdi_low"], d["hdi_high"] - d["mean"]], fmt="o", color="#4C72B0", label="posterior mean, 94% HDI"
        )
        ax.scatter(d["truth"], range(len(d)), marker="D", color="black", zorder=5, label="truth")
        ax.set_yticks(range(len(d)), [p.split("[")[1].rstrip("]") for p in d["parameter"]])
        ax.set_title(title)
        ax.legend(fontsize=7)
    d = rec[rec["group"] == "beta"]
    cols = [COLORS[p.split(",")[1].rstrip("]")] for p in d["parameter"]]
    ax = axes[2]
    ax.errorbar(d["truth"], d["mean"], yerr=[d["mean"] - d["hdi_low"], d["hdi_high"] - d["mean"]], fmt="none", ecolor="#bbbbbb", lw=0.8)
    ax.scatter(d["truth"], d["mean"], c=cols, s=18, zorder=3)
    lim = [0, max(d["hdi_high"].max(), d["truth"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("true beta (region x channel)")
    ax.set_ylabel("posterior mean")
    ax.set_title(f"Region-level effect ceilings: {int(d['in_hdi'].sum())}/{len(d)} in 94% HDI")
    for c in CHANNELS:
        ax.scatter([], [], color=COLORS[c], label=c, s=18)
    ax.legend(fontsize=7)
    _save(fig, path)


def contributions_plot(dates, baseline_k: np.ndarray, contrib_k: np.ndarray, path: Path) -> None:
    """contrib_k: (T, C) posterior mean national contribution."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(
        dates,
        baseline_k / 1e3,
        *(contrib_k[:, k] / 1e3 for k in range(len(CHANNELS))),
        labels=["baseline + controls", *CHANNELS],
        colors=["#dddddd", *(COLORS[c] for c in CHANNELS)],
        alpha=0.9,
    )
    ax.set_ylabel("$M / week")
    ax.set_title("Decomposition of national weekly sales (posterior mean, SYNTHETIC)")
    ax.legend(loc="upper left", fontsize=8, ncol=5)
    _save(fig, path)


def response_curves(rm, truth_rm, s0: np.ndarray, s_opt: np.ndarray, s_risk: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(15, 3.6))
    for k, (ax, c) in enumerate(zip(axes, CHANNELS, strict=True)):
        grid = np.linspace(0, 2.5 * s0[k], 60)
        draws = np.empty((rm.beta.shape[0], grid.size))
        true = np.empty(grid.size)
        for j, g in enumerate(grid):
            s = s0.copy()
            s[k] = g
            draws[:, j] = rm.channel_draws(s)[:, k]
            true[j] = truth_rm.channel_draws(s)[0, k]
        lo, hi = np.quantile(draws, Q, axis=0)
        ax.fill_between(grid, lo / 1e3, hi / 1e3, color=COLORS[c], alpha=0.3, label="94% interval")
        ax.plot(grid, draws.mean(axis=0) / 1e3, color=COLORS[c], label="posterior mean")
        ax.plot(grid, true / 1e3, "k--", lw=1, label="truth")
        for val, m, lab in ((s0[k], "o", "current"), (s_opt[k], "^", "expected-optimal"), (s_risk[k], "s", "risk-adjusted")):
            ax.axvline(val, color="grey", lw=0.6)
            ax.scatter([val], [np.interp(val, grid, draws.mean(axis=0)) / 1e3], marker=m, color="black", zorder=5, label=lab)
        ax.set_title(c)
        ax.set_xlabel("weekly spend ($k per million people)")
        if k == 0:
            ax.set_ylabel("weekly contribution ($M)")
            ax.legend(fontsize=6)
    fig.suptitle("Steady-state response curves (SYNTHETIC)", y=1.04)
    _save(fig, path)


def budget_plot(alloc: pd.DataFrame, lift_draws: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    keys = [k for k in ("current", "expected_optimal", "risk_adjusted", "last_click_guided") if k in alloc.columns]
    width = 0.8 / len(keys)
    x = np.arange(len(alloc))
    for i, k in enumerate(keys):
        ax.bar(x + i * width, alloc[k] / 1e3, width, color=ALLOC_COLORS[k], label=k.replace("_", " "))
    ax.set_xticks(x + width * (len(keys) - 1) / 2, alloc["channel"])
    ax.set_ylabel("annual spend ($M)")
    ax.set_title("Allocation of the same total budget")
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, d in lift_draws.items():
        ax.hist(d / 1e3, bins=60, alpha=0.5, density=True, color=ALLOC_COLORS[name], label=name.replace("_", " "))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("annual incremental sales vs current allocation ($M)")
    ax.set_title("Posterior lift distribution (SYNTHETIC)")
    ax.legend(fontsize=8)
    _save(fig, path)
