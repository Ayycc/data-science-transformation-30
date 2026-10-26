"""Static figures for results/ (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from claims_mlops import config as C  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2a6fdb", "#e8833a", "#8a8f98", "#c8423b"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def monitoring_timeline(mon: pd.DataFrame, cfg: C.RunConfig, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    m = mon.month
    ax = axes[0]
    for col, color in [("DrivAge", RED), ("Density", ORANGE), ("BonusMalus", BLUE), ("Area", GREY)]:
        ax.plot(m, mon[f"drift_{col}"], marker="o", ms=4, color=color, label=col)
    others = [c for c in C.FEATURES if c not in ("DrivAge", "Density", "BonusMalus", "Area")]
    ax.plot(m, mon[[f"drift_{c}" for c in others]].max(axis=1), ls=":", color="black", label="max of other 5 features")
    ax.axhline(0.1, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Evidently drift score\n(normed Wasserstein / JS)")
    ax.set_title("Monthly production batches vs reference (SYNTHETIC data)")
    ax.legend(fontsize=8, ncol=3)
    ax = axes[1]
    ax.plot(m, mon.ae_ratio, marker="o", color=BLUE, label="deployed champion (with retraining)")
    ax.plot(m, mon.static_ae_ratio, marker="s", color=GREY, label="initial champion, never retrained")
    lo, hi = 1 - cfg.trigger.ae_tolerance, 1 + cfg.trigger.ae_tolerance
    ax.axhspan(lo, hi, color=BLUE, alpha=0.06)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_ylabel("actual / expected cost")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(m, mon.deviance, marker="o", color=BLUE, label="deployed champion")
    ax.plot(m, mon.static_deviance, marker="s", color=GREY, label="initial champion, never retrained")
    ax.plot(m, mon.glm_deviance, marker="^", color=ORANGE, alpha=0.7, label="GLM v1 (reference)")
    ax.set_ylabel("Tweedie deviance (p=1.5)")
    ax.set_xlabel("month")
    ax.legend(fontsize=8)
    for a in axes:
        _style(a)
        a.axvline(cfg.sim.shift_start_month - 0.5, color=RED, lw=1, ls="--")
        for t in mon.loc[mon.trigger, "month"]:
            a.axvline(t + 0.5, color="green", lw=1, alpha=0.6)
    axes[0].text(cfg.sim.shift_start_month - 0.4, axes[0].get_ylim()[1] * 0.9, "covariate shift injected", color=RED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def drift_heatmap(
    mon: pd.DataFrame,
    out: Path,
    prefix: str = "fixed_drift_",
    flag_col: str = "fixed_ref_drifted_columns",
    title: str = "Per-feature Evidently drift score vs the fixed initial reference (SYNTHETIC)",
) -> None:
    mat = mon[[f"{prefix}{c}" for c in C.FEATURES]].to_numpy().T
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=max(0.3, float(mat.max())))
    ax.set_yticks(range(len(C.FEATURES)), C.FEATURES)
    ax.set_xticks(range(len(mon)), mon.month.astype(int))
    for i, c in enumerate(C.FEATURES):
        for j, flagged in enumerate(mon[flag_col].fillna("").str.split(";")):
            if c in flagged:
                ax.text(j, i, "x", ha="center", va="center", fontsize=8, color="white")
    ax.set_xlabel("month (x = flagged by Evidently)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def lift_chart(valid: pd.DataFrame, models: dict, out: Path, n_bins: int = 10) -> None:
    """Exposure-weighted decile lift: mean observed vs predicted pure premium by predicted decile."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    y, w = valid[C.TARGET].to_numpy(), valid[C.WEIGHT].to_numpy()
    for ax, (name, model) in zip(axes, models.items(), strict=True):
        p = model.predict(valid)
        order = np.argsort(p)
        cw = np.cumsum(w[order]) / w.sum()
        b = np.minimum((cw * n_bins).astype(int), n_bins - 1)
        obs = [np.average(y[order][b == k], weights=w[order][b == k]) for k in range(n_bins)]
        pred = [np.average(p[order][b == k], weights=w[order][b == k]) for k in range(n_bins)]
        x = np.arange(1, n_bins + 1)
        ax.bar(x, obs, color=GREY, alpha=0.6, label="observed")
        ax.plot(x, pred, marker="o", color=BLUE, label="predicted")
        ax.set_title(name)
        ax.set_xlabel("predicted decile (exposure-weighted)")
        _style(ax)
    axes[0].set_ylabel("pure premium (EUR / policy-year)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation months lift chart (SYNTHETIC)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
