"""Figures (matplotlib PNG + one interactive Plotly HTML). All data SYNTHETIC."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

COLORS = {
    "static": "#6b7280",
    "greedy_naive": "#b45309",
    "greedy_iv": "#d97706",
    "eps_greedy": "#0891b2",
    "linucb": "#2563eb",
    "thompson": "#7c3aed",
    "linucb_no_rm": "#93c5fd",
    "oracle": "#111827",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def regret_curves(runs: dict, names, path: Path, html: Path | None = None) -> None:
    seeds = sorted(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pfig = go.Figure()
    for n in names:
        if n == "oracle":
            continue
        cum = np.array([runs[s][n]["cum_regret"] for s in seeds]) / 1e3
        t = np.arange(1, cum.shape[1] + 1)
        m = cum.mean(0)
        for ax in axes:
            ax.plot(t, m, color=COLORS.get(n), label=n, lw=2 if n in ("linucb", "thompson", "eps_greedy") else 1.3)
            ax.fill_between(t, np.quantile(cum, 0.1, 0), np.quantile(cum, 0.9, 0), color=COLORS.get(n), alpha=0.12)
        pfig.add_trace(go.Scatter(x=t, y=m, name=n, line=dict(color=COLORS.get(n))))
    axes[0].set(title="Cumulative expected regret (mean, 10-90% band over seeds)", xlabel="night", ylabel="regret, $ thousands")
    axes[1].set(xscale="log", yscale="log", title="Log-log: slope < 1 means sublinear", xlabel="night (log)", ylabel="regret (log)")
    tt = np.arange(30, cum.shape[1] + 1)
    axes[1].plot(tt, tt * (axes[1].get_ylim()[1] / cum.shape[1]) * 0.25, ls="--", color="k", lw=0.8, label="slope 1 reference")
    for ax in axes:
        _style(ax)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("SYNTHETIC hotel market: regret vs the best guardrail-feasible price under the true demand curve", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    if html is not None:
        pfig.update_layout(
            title="Cumulative expected regret, mean over seeds (SYNTHETIC)", xaxis_title="night", yaxis_title="$ thousands", template="plotly_white"
        )
        pfig.write_html(html, include_plotlyjs="cdn")


def uplift_ci(P: dict, path: Path) -> None:
    names = [n for n in P if n != "static"]
    m = np.array([P[n]["uplift_vs_static"]["mean"] for n in names]) * 100
    lo = np.array([P[n]["uplift_vs_static"]["t_ci"][0] for n in names]) * 100
    hi = np.array([P[n]["uplift_vs_static"]["t_ci"][1] for n in names]) * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(names))
    ax.barh(y, m, color=[COLORS.get(n) for n in names], alpha=0.85)
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor="k", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for yi, mi, hii in zip(y, m, hi):
        ax.text(max(hii, 0) + 0.4, yi, f"{mi:+.2f}%", va="center", ha="left", fontsize=8)
    ax.set_xlim(right=max(hi.max(), 0) + 6)
    ax.set(xlabel="room revenue uplift vs static table, % (95% CI over seeds)", title="Revenue uplift vs static rate table (SYNTHETIC)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def price_paths(runs: dict, guard, path: Path, bucket: int = 1, n_nights: int = 150) -> None:
    s = sorted(runs)[0]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for n in ("static", "greedy_iv", "thompson", "oracle"):
        if n in runs[s]:
            ax.plot(runs[s][n]["prices"][:n_nights, bucket], color=COLORS[n], label=n, lw=1.6 if n != "static" else 2.2)
    ax.axhline(guard.floor, color="r", ls=":", lw=1)
    ax.axhline(guard.ceiling, color="r", ls=":", lw=1, label=f"floor \\${guard.floor:.0f} / ceiling \\${guard.ceiling:.0f}")
    ax.set(
        xlabel="night",
        ylabel="posted rate, $",
        title=f"Posted rates, booking window {bucket} (seed {s}); max change {guard.max_change:.0%} per night (SYNTHETIC)",
    )
    ax.legend(fontsize=8, ncol=5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def elasticity(E: dict, path: Path) -> None:
    keys = [("naive_ols", "naive OLS (all cells)"), ("randomised_test_ols", "OLS, randomised cells"), ("iv_2sls", "2SLS, z instrument")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    tr = E["truth"]
    ax.axvspan(tr["theta_p10"], tr["theta_p90"], color="#10b981", alpha=0.12, label="truth p10-p90 across cells")
    ax.axvline(tr["theta_mean"], color="#10b981", lw=2, label=f"truth mean {tr['theta_mean']:.2f}")
    for i, (k, _lab) in enumerate(keys):
        th, (lo, hi) = E[k]["theta"], E[k]["ci95"]
        ax.errorbar(th, i, xerr=[[th - lo], [hi - th]], fmt="o", color="k", capsize=4)
        ax.annotate(f"{th:.2f}", (hi, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(keys)), [lab for _, lab in keys])
    ax.invert_yaxis()
    ax.set(xlabel="semi-elasticity d log demand / d (price / static rate)", title="Price sensitivity from the logged year (SYNTHETIC)")
    ax.legend(fontsize=7, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def revpar_volatility(P: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n, v in P.items():
        ax.scatter(v["volatility_mean_abs_dlog"] * 100, v["revpar"]["mean"], color=COLORS.get(n), s=60)
        ax.annotate(n, (v["volatility_mean_abs_dlog"] * 100, v["revpar"]["mean"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="price volatility: mean |night-over-night change|, %", ylabel="RevPAR, $", title="RevPAR vs price volatility (SYNTHETIC)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(runs: dict, metrics: dict, logs, scfg, outdir: Path) -> None:
    P = metrics["study"]["policies"]
    regret_curves(runs, scfg.policies, outdir / "regret_curves.png", outdir / "regret_curves.html")
    uplift_ci(P, outdir / "uplift_vs_static.png")
    price_paths(runs, scfg.guard, outdir / "price_paths.png")
    elasticity(metrics["elasticity"], outdir / "elasticity_estimates.png")
    revpar_volatility(P, outdir / "revpar_vs_volatility.png")
