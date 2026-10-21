"""Static figures (matplotlib, Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8a8a85"
INK = "#2b2b29"
INK2 = "#5f5e5a"
GRID = "#e4e3de"
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#13396b"])
STRAT_COLOR = {"random": MUTED, "margin": SERIES[0], "entropy": SERIES[2], "margin_diverse": SERIES[1]}
STRAT_LABEL = {
    "random": "Random sampling",
    "margin": "Margin (uncertainty)",
    "entropy": "Entropy (uncertainty)",
    "margin_diverse": "Margin + k-means diversity",
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def learning_curves(curves: dict, pool_size: int, title: str, path: Path, target: float | None = None) -> None:
    """curves: {strategy: (n_labels list, mean f1, std f1)}."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for s, (n, m, sd) in curves.items():
        x = np.asarray(n) / pool_size * 100
        ax.plot(x, m, color=STRAT_COLOR.get(s, INK), lw=2, label=STRAT_LABEL.get(s, s))
        ax.fill_between(x, np.asarray(m) - sd, np.asarray(m) + sd, color=STRAT_COLOR.get(s, INK), alpha=0.12, lw=0)
    if target is not None:
        ax.axhline(target, color=INK2, lw=1, ls="--")
        ax.text(0.02, target + 0.01, f"random @ 2% budget = {target:.3f}", color=INK2, fontsize=8, transform=ax.get_yaxis_transform())
        ax.axvline(1.0, color=GRID, lw=1.5)
        ax.text(1.02, 0.05, "50% of budget", color=INK2, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Labelled share of the pool (%)", color=INK2)
    ax.set_ylabel("Test macro-F1", color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(cm: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7, color=INK2)
    ax.set_yticks(range(len(labels)), labels, fontsize=7, color=INK2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0 and (i == j or norm[i, j] >= 0.02):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5.5, color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted intent", color=INK2)
    ax.set_ylabel("True intent", color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_intent_f1(rows: dict[str, dict[str, float]], path: Path) -> None:
    """rows: {model_label: {intent: f1}} for at most 3 models."""
    models = list(rows)
    intents = sorted(next(iter(rows.values())), key=lambda k: rows[models[-1]][k])
    y = np.arange(len(intents))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(intents) + 1.5))
    for k, m in enumerate(models):
        ax.barh(y + k * h, [rows[m][i] for i in intents], height=h * 0.9, color=SERIES[k], label=m)
    ax.set_yticks(y + h * (len(models) - 1) / 2, intents, fontsize=7.5, color=INK2)
    ax.set_xlabel("Test F1", color=INK2)
    ax.set_xlim(0, 1.0)
    ax.set_title("Per-intent F1", color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(models))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def abstention_curve(curves: dict[str, tuple[np.ndarray, np.ndarray]], points: dict[str, tuple[float, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for k, (name, (cov, acc)) in enumerate(curves.items()):
        ax.plot(cov * 100, acc, color=SERIES[k], lw=2, label=name)
        if name in points:
            c, a = points[name]
            ax.plot([c * 100], [a], "o", ms=8, color=SERIES[k], mec="white", mew=2)
            ax.annotate(
                f"{c * 100:.1f}% auto-routed, acc {a:.3f}",
                (c * 100, a),
                xytext=(-150, -25),
                textcoords="offset points",
                fontsize=8,
                color=INK2,
            )
    ax.set_xlabel("Tickets auto-routed (coverage, %)", color=INK2)
    ax.set_ylabel("Accuracy on auto-routed tickets", color=INK2)
    ax.set_title("Calibrated abstention: low-confidence tickets go to humans", color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def umap_scatter(U2: np.ndarray, clusters: np.ndarray, highlight: np.ndarray, highlight_names: list[str], path: Path) -> None:
    """U2: 2-D UMAP; highlight: int array (-1 = other, k = highlight_names[k])."""
    fig, ax = plt.subplots(figsize=(7.2, 6))
    base = highlight == -1
    noise = clusters == -1
    ax.scatter(U2[base & ~noise, 0], U2[base & ~noise, 1], s=3, color="#b9b8b2", lw=0, label="known intents (clustered)")
    ax.scatter(U2[base & noise, 0], U2[base & noise, 1], s=3, color="#e4e3de", lw=0, label="HDBSCAN noise")
    for k, name in enumerate(highlight_names):
        m = highlight == k
        ax.scatter(U2[m, 0], U2[m, 1], s=6, color=SERIES[1 + k], lw=0, label=f"held-out: {name}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("Test tickets in 2-D UMAP; held-out 'emerging' intents highlighted", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, markerscale=3, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pii_bars(per_type: dict[str, dict[str, float]], path: Path) -> None:
    """per_type: {variant: {type: recall}}."""
    variants = list(per_type)
    types = sorted(next(iter(per_type.values())))
    x = np.arange(len(types))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for k, v in enumerate(variants):
        vals = [per_type[v][t] for t in types]
        ax.bar(x + k * w, vals, width=w * 0.9, color=SERIES[k], label=v)
    ax.axhline(0.95, color=INK2, lw=1, ls="--")
    ax.text(len(types) - 0.5, 0.955, "95% target", color=INK2, fontsize=8, ha="right")
    ax.set_xticks(x + w * (len(variants) - 1) / 2, types, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Span recall", color=INK2)
    ax.set_title("PII redaction recall by type (generator-labelled set)", color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
