"""Static figures (matplotlib, Agg). Palette: validated reference categorical slots in fixed order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(w=7.5, h=4.2, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    for ax in np.atleast_1d(axes):
        _style(ax)
    return fig, axes


def degree_ccdf(indeg: np.ndarray, outdeg: np.ndarray, path: Path) -> None:
    fig, ax = _fig()
    for i, (name, d) in enumerate((("in-degree (distinct senders)", indeg), ("out-degree (distinct receivers)", outdeg))):
        d = np.sort(d[d > 0])
        ccdf = 1.0 - np.arange(len(d)) / len(d)
        ax.loglog(d, ccdf, color=SERIES[i], linewidth=2, label=name)
    ax.set_xlabel("degree", color=INK2)
    ax.set_ylabel("share of accounts with degree >= x", color=INK2)
    ax.set_title("Degree distribution (CCDF), SYNTHETIC transaction graph", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def weekly(snap: pd.DataFrame, path: Path) -> None:
    fig, axes = _fig(w=10, h=3.6, ncols=2)
    axes[0].plot(snap["week"], snap["transactions"] / 1e3, color=SERIES[0], linewidth=2)
    axes[0].set_title("Transactions per week (thousands)", color=INK, fontsize=10, loc="left")
    axes[1].plot(snap["week"], snap["laundering_tx"], color=SERIES[1], linewidth=2)
    axes[1].set_title("Injected laundering transfers per week", color=INK, fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("week", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], ap: dict[str, float], path: Path) -> None:
    fig, ax = _fig(w=7, h=5)
    for i, (name, s) in enumerate(scores.items()):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=SERIES[i], linewidth=2, label=f"{name} (PR-AUC {ap[name]:.3f})")
    ax.axhline(y.mean(), color=INK2, linewidth=1, linestyle="--", label=f"base rate {y.mean():.3f}")
    ax.set_xlabel("recall", color=INK2)
    ax.set_ylabel("precision", color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_title("Held-out PR curves: tabular vs graph feature sets", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def typology_recall(rules: dict[str, float], model: dict[str, float], path: Path) -> None:
    typs = list(rules)
    x = np.arange(len(typs))
    fig, ax = _fig(w=8, h=4.2)
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, [rules[t] for t in typs], w, color=SERIES[0], label="motif detector (instance recall, all instances)")
    b2 = ax.bar(
        x + w / 2 + 0.01, [model.get(t, 0) for t in typs], w, color=SERIES[1], label="graph model @80% recall (account recall, test)"
    )
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK2,
            )
    ax.set_xticks(x, [t.replace("_", " ") for t in typs])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("recall", color=INK2)
    ax.set_title("Recall per injected typology", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def importance(imp: pd.Series, path: Path, k: int = 15) -> None:
    top = imp.sort_values(ascending=True).tail(k)
    fig, ax = _fig(w=7.5, h=5)
    ax.barh(top.index, top.to_numpy(), color=SERIES[0], height=0.7)
    ax.set_xlabel("mean |SHAP| on test accounts (log-odds)", color=INK2)
    ax.set_title("Top features of the graph model", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def workload(rows: pd.DataFrame, path: Path) -> None:
    fig, ax = _fig(w=7.5, h=3.8)
    ax.barh(rows["feature_set"], rows["false_positives"], color=SERIES[0], height=0.6)
    for i, v in enumerate(rows["false_positives"]):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=8.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("false alerts on test accounts at 80% recall", color=INK2)
    ax.set_title("Investigator workload at equal recall", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def subgraph_png(g: nx.MultiDiGraph, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    simple = nx.DiGraph(g)
    pos = nx.spring_layout(simple, seed=7, k=0.9)
    focal = g.graph["focal"]
    colors = [SERIES[1] if n == focal else "#9a9893" for n in simple.nodes]
    motif = [(u, v) for u, v, d in g.edges(data=True) if d["motif"]]
    other = [(u, v) for u, v, d in g.edges(data=True) if not d["motif"]]
    nx.draw_networkx_edges(simple, pos, edgelist=other, edge_color="#c3c2b7", width=1, ax=ax, arrows=True, arrowsize=8)
    nx.draw_networkx_edges(simple, pos, edgelist=motif, edge_color="#e34948", width=2.2, ax=ax, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(
        simple, pos, node_color=colors, node_size=[420 if n == focal else 140 for n in simple.nodes], ax=ax, edgecolors=SURFACE
    )
    nx.draw_networkx_labels(simple, pos, labels={focal: str(focal)}, font_size=8, ax=ax)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
