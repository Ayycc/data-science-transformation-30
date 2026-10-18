"""Figures for results/ (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def reset_style() -> None:
    """Restore a neutral style (causalml switches matplotlib to 'fivethirtyeight' on import)."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


reset_style()


COLORS = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644"]
SYN = "SYNTHETIC data"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def km_panels(panels: list[tuple[str, dict, float]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
    for ax, (title, curves, p) in zip(np.atleast_1d(axes), panels):
        for i, (name, km) in enumerate(curves.items()):
            km.plot_survival_function(ax=ax, ci_show=True, color=COLORS[i % len(COLORS)], label=name)
        ax.set_title(f"{title} (log-rank p{'<1e-300' if p == 0 else f'={p:.1e}'})")
        ax.set_xlabel("tenure (months)")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("S(t): share still subscribed")
    fig.suptitle(f"Kaplan-Meier survival by segment ({SYN})", y=1.02)
    _save(fig, path)


def schoenfeld_plot(res: pd.DataFrame, covariates: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(covariates), figsize=(5 * len(covariates), 3.8))
    for ax, c in zip(np.atleast_1d(axes), covariates):
        g = res.groupby("duration")[c].agg(["mean", "count"])
        ax.scatter(res["duration"] + np.random.default_rng(0).uniform(-0.3, 0.3, len(res)), res[c], s=2, alpha=0.08, color="#888")
        ax.plot(g.index, g["mean"], color=COLORS[1], lw=2, label="monthly mean")
        ax.axhline(res[c].mean(), color=COLORS[0], ls="--", label="constant effect (PH)")
        ax.set_ylim(np.percentile(res[c], 1), np.percentile(res[c], 99))
        ax.set_title(f"scaled Schoenfeld: {c}")
        ax.set_xlabel("tenure (months)")
        ax.legend(fontsize=8)
    _save(fig, path)


def td_auc_plot(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, m) in enumerate(metrics.items()):
        t = list(m["td_auc"].keys())
        ax.plot(t, list(m["td_auc"].values()), marker="o", color=COLORS[i], label=f"{name} (Uno C={m['uno_c']:.3f})")
    ax.set_xlabel("horizon t (months)")
    ax.set_ylabel("cumulative/dynamic AUC(t)")
    ax.set_title(f"Time-dependent AUC on test customers ({SYN})")
    ax.legend()
    _save(fig, path)


def cif_plot(aj: pd.DataFrame, causes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(causes):
        ax.plot(aj["month"], aj[f"cif_{c}"], color=COLORS[i], label=c)
    ax.plot(aj["month"], 1 - aj["overall_survival"], color="k", ls="--", label="any churn")
    ax.set_xlabel("tenure (months)")
    ax.set_ylabel("cumulative incidence")
    ax.set_title(f"Competing risks: Aalen-Johansen CIF ({SYN})")
    ax.legend()
    _save(fig, path)


def clv_plot(clv_hat: np.ndarray, clv_true: np.ndarray, contract: np.ndarray, labels: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(clv_hat), min(6000, len(clv_hat)), replace=False)
    for i, lab in enumerate(labels):
        m = contract[idx] == i
        axes[0].scatter(clv_true[idx][m], clv_hat[idx][m], s=3, alpha=0.3, color=COLORS[i], label=lab)
    lim = [0, np.percentile(np.r_[clv_true, clv_hat], 99.5)]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("true 36-month CLV ($, from generator)")
    axes[0].set_ylabel("CLV from stratified Cox survival curve ($)")
    axes[0].legend(markerscale=4)
    for i, lab in enumerate(labels):
        axes[1].hist(clv_hat[contract == i], bins=60, alpha=0.55, color=COLORS[i], label=lab)
    axes[1].set_xlabel("model CLV ($)")
    axes[1].set_title("CLV distribution by contract")
    fig.suptitle(f"Customer lifetime value at campaign time ({SYN})")
    _save(fig, path)


def qini_plot(curves: dict, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (name, (f, q)) in enumerate(curves.items()):
        ls = ":" if name in ("random", "oracle (true uplift)") else "-"
        ax.plot(f, q, ls=ls, color=COLORS[i % len(COLORS)] if ls == "-" else "k", label=name)
    ax.set_xlabel("fraction of customers targeted")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def uplift_by_decile(rows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    x = np.arange(len(rows))
    ax.bar(x - w / 2, rows["observed_uplift"], w, color=COLORS[0], label="observed (treated - control retention)")
    ax.bar(x + w / 2, rows["true_uplift"], w, color=COLORS[1], label="true mean uplift")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("predicted-uplift decile (1 = highest)")
    ax.set_ylabel("6-month retention uplift")
    ax.set_title(f"Holdout uplift by decile, selected model ({SYN})")
    ax.legend(fontsize=8)
    _save(fig, path)


def policy_bars(tab: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    t = tab.sort_values("net_incremental_revenue_true")
    ax.barh(
        t["policy"],
        t["net_incremental_revenue_true"] / 1e3,
        color=[COLORS[1] if v < 0 else COLORS[0] for v in t["net_incremental_revenue_true"]],
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("true net incremental revenue on holdout ($k, 36-month horizon)")
    ax.set_title(f"Targeting policies under the same budget ({SYN})")
    _save(fig, path)


def budget_curve(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (name, g) in enumerate(df.groupby("policy", sort=False)):
        ax.plot(g["budget"] / 1e3, g["net_incremental_revenue_true"] / 1e3, marker="o", color=COLORS[i], label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("campaign budget ($k)")
    ax.set_ylabel("true net incremental revenue ($k)")
    ax.set_title(f"Value vs budget on holdout ({SYN})")
    ax.legend(fontsize=8)
    _save(fig, path)
