"""Static figures (matplotlib, PNG). Colours: fixed categorical order from a validated colour-blind-safe palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def _line(ax, x, y, i: int, label: str, ls: str = "-", marker: str = "o") -> None:
    ax.plot(x, y, ls, color=C[i], lw=2, marker=marker, ms=6, mec="white", mew=1.5, label=label)


def make_figures(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb"})

    # 1. accuracy vs epsilon for the aggregate statistics
    cur = res["accuracy_curves"]
    e = cur["epsilon"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    _line(ax, e, cur["count_rel_err_white"], 0, "White cells, Laplace")
    _line(ax, e, cur["count_rel_err_rare"], 1, "Amer-Indian-Eskimo cells, Laplace")
    _line(ax, e, cur["count_rel_err_white_gauss"], 0, "White cells, Gaussian (zCDP, δ=1e-6)", ls="--", marker="s")
    _line(ax, e, cur["count_rel_err_rare_gauss"], 1, "Amer-Indian-Eskimo cells, Gaussian", ls="--", marker="s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Counts race × sex × income: mean relative error", "epsilon", "mean |error| / true count")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    _line(ax, e, cur["age_hist_tvd"], 0, "age histogram (14 bins)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Age histogram: total variation distance", "epsilon", "TVD to the true histogram")
    ax = axes[2]
    _line(ax, e, cur["mean_hours_abs_err_white"], 0, "White")
    _line(ax, e, cur["mean_hours_abs_err_rare"], 1, "Amer-Indian-Eskimo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Mean hours/week by race: absolute error", "epsilon", "mean |error| (hours)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_epsilon.png", dpi=130)
    plt.close(fig)

    # 2. composition: Laplace/basic vs Gaussian/zCDP
    comp = res["composition"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _line(ax, comp["k"], comp["laplace_std"], 0, "Laplace, basic composition")
    _line(ax, comp["k"], comp["gaussian_zcdp_std"], 1, "Gaussian, zCDP accounting")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(
        ax, f"Noise per count when ε={comp['epsilon']:g}, δ={comp['delta']:g} is split over k queries", "number of queries k", "noise std per answer"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "composition_zcdp.png", dpi=130)
    plt.close(fig)

    # 3. DP model
    dm = res["dp_model"]
    eps = [r["epsilon"] for r in dm["dp"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, key, title in [(axes[0], "auc", "ROC AUC on real holdout"), (axes[1], "accuracy", "Accuracy on real holdout")]:
        _line(ax, eps, [r[key] for r in dm["dp"]], 0, "DP model, all test records")
        _line(ax, eps, [r[f"{key}_rare"] for r in dm["dp"]], 1, f"DP model, Amer-Indian-Eskimo (n={dm['n_test_rare']})")
        ax.axhline(dm["baseline"][key], color=C[0], lw=1.2, ls=":", label="non-private, all")
        ax.axhline(dm["baseline"][f"{key}_rare"], color=C[1], lw=1.2, ls=":", label="non-private, Amer-Indian-Eskimo")
        if key == "accuracy":
            ax.axhline(dm["majority_class_accuracy"], color=INK2, lw=1, ls="--", label="always predict <=50K")
        ax.set_xscale("log")
        _style(ax, f"DP logistic regression: {title}", "epsilon", key)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "dp_model_vs_epsilon.png", dpi=130)
    plt.close(fig)

    # 4. privacy-utility frontier
    rels = res["releases"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, xkey, xlabel in [
        (axes[0], lambda r: r["mia_dcr"]["auc"], "membership-inference AUC (DCR attack; 0.5 = no signal)"),
        (axes[1], lambda r: r["anonymeter"]["max_risk"], "max Anonymeter risk (singling out, linkability, inference)"),
    ]:
        dp = [r for r in rels if r.get("formal_dp")]
        ax.plot([xkey(r) for r in dp], [r["tstr"]["auc"] for r in dp], "-", color=C[0], lw=2, zorder=1)
        for r in rels:
            if r.get("formal_dp"):
                col, mk, lab = C[0], "o", f"DP ε={r['epsilon']:g}"
            elif r["name"].startswith("Raw"):
                col, mk, lab = INK2, "X", "raw microdata"
            elif r["name"].startswith("k-anon"):
                col, mk, lab = C[3], "D", "k-anonymised (k=5)"
            elif "copula" in r["name"]:
                col, mk, lab = C[1], "s", "Gaussian copula"
            else:
                col, mk, lab = C[2], "^", "CTGAN-style"
            x, y = xkey(r), r["tstr"]["auc"]
            ax.scatter([x], [y], color=col, marker=mk, s=70, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK)
        _style(ax, "Privacy-utility frontier", xlabel, "TSTR ROC AUC on real holdout")
    axes[0].axvline(res["config"]["max_mia_auc"], color=INK2, lw=1, ls="--")
    axes[1].axvline(res["config"]["max_attack_risk"], color=INK2, lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(out / "privacy_utility_frontier.png", dpi=130)
    plt.close(fig)

    # 5. rare subgroup: TSTR accuracy loss and conditional fidelity per release
    names = [r["name"].split(" (")[0] if not r.get("formal_dp") else f"DP ε={r['epsilon']:g}" for r in rels]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    x = np.arange(len(rels))
    w = 0.38
    ax = axes[0]
    ax.bar(x - w / 2, [r["tstr"]["auc_other"] for r in rels], w * 0.95, color=C[0], label="all other records")
    ax.bar(x + w / 2, [r["tstr"]["auc_rare"] for r in rels], w * 0.95, color=C[1], label=f"Amer-Indian-Eskimo (n={rels[0]['tstr']['n_test_rare']})")
    ax.set_ylim(0.5, 0.95)
    ax.set_xticks(x, names, rotation=25, ha="right")
    _style(ax, "TSTR AUC: majority vs rare subgroup", "", "ROC AUC")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.bar(x - w / 2, [r["rare_subgroup"]["conditional_tvd_majority"] for r in rels], w * 0.95, color=C[0], label="White (majority)")
    ax.bar(x + w / 2, [r["rare_subgroup"]["conditional_tvd_rare"] for r in rels], w * 0.95, color=C[1], label="Amer-Indian-Eskimo")
    ax.set_xticks(x, names, rotation=25, ha="right")
    _style(ax, "Within-group marginal distortion (mean TVD)", "", "mean TVD of the group's 1-way marginals")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "rare_subgroup.png", dpi=130)
    plt.close(fig)

    # 6. budget ledger
    led = res["budget"]["ledger"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    left = 0.0
    for i, s in enumerate(led):
        ax.barh([0], [s["epsilon"]], left=left, color=C[i % len(C)], edgecolor="white", linewidth=2, label=f"{s['label']} ({s['epsilon']:g})")
        left += s["epsilon"]
    ax.barh([0], [res["budget"]["epsilon_total"] - left], left=left, color=GRID, edgecolor="white", linewidth=2, label="unspent reserve")
    ax.set_yticks([])
    ax.set_xlim(0, res["budget"]["epsilon_total"])
    _style(ax, f"Production privacy budget: ε spent {left:g} of {res['budget']['epsilon_total']:g}", "", "")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.6), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "budget_ledger.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
