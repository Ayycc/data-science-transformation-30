"""Static figures (matplotlib) written to results/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .ope import ESTIMATORS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BASELINE = "#a9a8a2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
EST_COLOR = dict(zip(ESTIMATORS, SERIES, strict=False))

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    }
)

OURS = ("Two-tower (exact)", "Two-tower (HNSW)", "Two-tower + re-ranker", "Two-tower + re-ranker (+ALS feature)")


def _bars(ax, names, values, fmt, title, ref=None, ref_label=None):
    y = np.arange(len(names))[::-1]
    colors = [SERIES[0] if n in OURS else BASELINE for n in names]
    ax.barh(y, values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", color=INK, fontsize=8)
    ax.set_xlim(0, max([*values, ref or 0]) * 1.22)
    if ref is not None:
        ax.axvline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.text(ref, -0.75, f" {ref_label}", color=INK2, fontsize=8, ha="left", va="center")
    ax.set_ylim(-1.1, len(names) - 0.4)  # same vertical layout in every panel


def offline_metrics(res: dict, path: Path) -> None:
    t = res["offline"]["test"]
    names = list(t)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    _bars(axes[0], names, [t[n]["ndcg@10"] for n in names], "{:.4f}", "NDCG@10 (test)")
    _bars(axes[1], names, [t[n]["recall@10"] for n in names], "{:.4f}", "Recall@10 (test)")
    _bars(axes[2], names, [t[n]["recall@50"] for n in names], "{:.4f}", "Recall@50 (test)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle(f"Offline ranking quality on held-out next titles ({res['data']['label']} data)", x=0.01, y=1.06, ha="left", fontsize=12, color=INK)
    fig.savefig(path)
    plt.close(fig)


def popularity_bias(res: dict, path: Path) -> None:
    t = res["offline"]["test"]
    ref = res["offline"]["test_reference"]
    names = list(t)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    _bars(
        axes[0],
        names,
        [100 * t[n]["head_share@10"] for n in names],
        "{:.1f}%",
        "Head share@10 (%): top-10% titles",
        100 * ref["head_share"],
        "what users watched next",
    )
    _bars(axes[1], names, [100 * t[n]["coverage@10"] for n in names], "{:.1f}%", "Catalogue coverage@10 (%)")
    _bars(axes[2], names, [t[n]["novelty@10"] for n in names], "{:.2f}", "Novelty@10 (bits)", ref["novelty"], "next titles")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Popularity bias: how concentrated each model's recommendations are", x=0.01, y=1.06, ha="left", fontsize=12, color=INK)
    fig.savefig(path)
    plt.close(fig)


def ope_errors(ope_art: dict, path: Path) -> None:
    main, truth, names = ope_art["main"], ope_art["truth"], ope_art["names"]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    width = 0.18
    for j, e in enumerate(ESTIMATORS):
        data = [100 * (main[n][e] - truth[n]) / truth[n] for n in names]
        pos = np.arange(len(names)) + (j - 1.5) * width
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.2},
            whiskerprops={"color": INK2},
            capprops={"color": INK2},
        )
        for b in bp["boxes"]:
            b.set_facecolor(EST_COLOR[e])
            b.set_edgecolor(SURFACE)
        ax.plot([], [], color=EST_COLOR[e], linewidth=8, label=e)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(np.arange(len(names)), [n.replace(" + ", " +\n") for n in names])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("estimate - truth (% of true play rate)")
    ax.set_title(f"Off-policy estimates vs simulated truth across {len(main[names[0]]['DR'])} independent logs (whiskers: 1.5 IQR)", loc="left")
    ax.legend(ncol=4, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def ope_rmse_vs_n(res: dict, path: Path) -> None:
    sweep = res["ope"]["rmse_vs_rounds"]
    truth = res["ope"]["truth"]
    ns = [int(n) for n in sweep]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    all_vals = []
    for e in ESTIMATORS:
        vals = [100 * np.mean([sweep[str(n)][p][e] / truth[p] for p in sweep[str(n)]]) for n in ns]
        all_vals += vals
        ax.plot(ns, vals, color=EST_COLOR[e], linewidth=2, marker="o", markersize=5, label=e)
        ax.text(ns[-1] * 1.08, vals[-1], e, color=INK, va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    yt = [v for v in (0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 30) if min(all_vals) * 0.85 <= v <= max(all_vals) * 1.15]
    ax.set_yticks(yt, [f"{v}%" for v in yt])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("logged rounds")
    ax.set_ylabel("relative RMSE, mean over policies")
    ax.set_title("Estimator error vs log size", loc="left")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def latency(res: dict, samples: list[float], path: Path) -> None:
    lat = res["offline"]["latency"]["hnsw"]["end_to_end"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    s = np.asarray(samples)
    ax.hist(np.clip(s, None, np.percentile(s, 99.8)), bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    for key, lab in (("p50_ms", "p50"), ("p99_ms", "p99")):
        ax.axvline(lat[key], color=INK, linestyle="--", linewidth=1)
        ax.text(lat[key], ax.get_ylim()[1] * 0.92, f" {lab} {lat[key]:.2f} ms", color=INK, fontsize=8)
    ax.set_xlabel("ms per request (user tower + HNSW top-100, single thread)")
    ax.set_ylabel("requests")
    ax.grid(axis="x", visible=False)
    ax.set_title("Retrieval latency", loc="left")
    fig.savefig(path)
    plt.close(fig)


def mmr_tradeoff(res: dict, path: Path) -> None:
    mm = res["offline"]["mmr"]
    lams = list(mm)
    x = [mm[k]["ild@10"] for k in lams]
    y = [mm[k]["ndcg@10"] for k in lams]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(x, y, color=SERIES[0], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, yi, k in zip(x, y, lams, strict=True):
        ax.annotate(f"λ={k}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax.set_xlabel("intra-list genre diversity of top-10")
    ax.set_ylabel("NDCG@10")
    ax.set_title("MMR re-ranking: relevance vs diversity", loc="left")
    fig.savefig(path)
    plt.close(fig)


def make_all(res: dict, art: dict, ope_art: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    offline_metrics(res, out / "offline_metrics.png")
    popularity_bias(res, out / "popularity_bias.png")
    ope_errors(ope_art, out / "ope_estimates_vs_truth.png")
    ope_rmse_vs_n(res, out / "ope_rmse_vs_rounds.png")
    latency(res, art["latency_samples"], out / "retrieval_latency.png")
    mmr_tradeoff(res, out / "mmr_tradeoff.png")
