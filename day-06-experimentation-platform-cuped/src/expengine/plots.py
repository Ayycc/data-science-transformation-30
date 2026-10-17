"""Figures for the README (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
C = {"naive_peeking": "#d64545", "msprt": "#2f6fdf", "msprt_cuped": "#0f9d8a", "bayes_peeking_p95": "#e8a33d"}
NAMES = {
    "naive_peeking": "Naive z-test, peek daily",
    "msprt": "mSPRT always-valid",
    "msprt_cuped": "mSPRT + CUPED",
    "bayes_peeking_p95": "Bayes P(T>C) >= 0.95, peek daily",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e4e7eb", lw=0.8)
    ax.set_axisbelow(True)


def fpr_curves(mc_aa: dict, path: Path, alpha: float = 0.05) -> None:
    metrics = list(mc_aa)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        curves = mc_aa[m]["rejection_curve"]
        looks = np.arange(1, len(curves["msprt"]) + 1)
        for k, v in curves.items():
            ax.plot(looks, v, color=C[k], lw=2, label=NAMES[k])
        ax.axhline(alpha, color=MUTED, ls="--", lw=1)
        ax.set_title(m, color=INK, fontsize=10)
        ax.set_xlabel("daily look", color=INK, fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("cumulative false-positive rate (A/A)", color=INK, fontsize=9)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Peeking every day inflates false positives; mSPRT stays under 5% (SYNTHETIC A/A)", color=INK, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cuped_bars(rows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(rows))
    ax.bar(x, rows["vr_empirical"] * 100, color="#2f6fdf", width=0.55)
    for i, r in rows.reset_index(drop=True).iterrows():
        ax.text(
            i,
            r["vr_empirical"] * 100 + 1.5,
            f"{r['vr_empirical'] * 100:.1f}%\nrho={r['rho']:.2f}",
            ha="center",
            fontsize=8.5,
            color=INK,
        )
    ax.axhline(30, color=MUTED, ls="--", lw=1)
    ax.text(len(rows) - 0.5, 31, "30% target", color=MUTED, fontsize=8, ha="right")
    ax.set_xticks(x, rows["metric"])
    ax.set_ylabel("variance reduction of the effect estimate", color=INK, fontsize=9)
    ax.set_ylim(0, max(80, rows["vr_empirical"].max() * 100 + 15))
    ax.set_title("CUPED variance reduction across Monte Carlo replicates (SYNTHETIC)", color=INK, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def novelty_curve(curve: pd.DataFrame, path: Path, title: str) -> None:
    c = curve.pivot(index="exposure_day", columns="variant", values="mean_sessions")
    lift = (c["treatment"] / c["control"] - 1) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(lift.index, lift.values, marker="o", color="#2f6fdf", lw=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("days since assignment", color=INK, fontsize=9)
    ax.set_ylabel("sessions/day lift (%)", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sequential_traces(readouts: list[dict], path: Path, alpha: float = 0.05) -> None:
    fig, axes = plt.subplots(1, len(readouts), figsize=(3.4 * len(readouts), 3.3), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), readouts):
        tr = r["trace"]
        days = np.arange(1, len(tr["p_naive"]) + 1)
        ax.plot(days, tr["p_naive"], color=C["naive_peeking"], lw=1.8, label="naive p (daily)")
        ax.plot(days, tr["p_always_valid"], color=C["msprt"], lw=2, label="always-valid p")
        ax.axhline(alpha, color=MUTED, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.5)
        ax.set_title(f"{r['exp_id']} {r['name']}\n({r['primary_metric']}, CUPED)", color=INK, fontsize=9)
        ax.set_xlabel("enrollment day", color=INK, fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("p-value (log scale)", color=INK, fontsize=9)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
