"""Static figures (matplotlib, Agg). Palette: validated categorical slots in fixed order; gray for reference lines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
FAMILY_COLOR = {
    "baseline": MUTED,
    "challenger": VIOLET,
    "candidate": BLUE,
    "lda_candidate": AQUA,
    "reweighing": ORANGE,
    "exponentiated_gradient": YELLOW,
    "threshold_optimizer": MAGENTA,
}
FAMILY_LABEL = {
    "baseline": "WOE scorecard",
    "challenger": "GBM (unconstrained)",
    "candidate": "GBM monotone (candidate)",
    "lda_candidate": "LDA search",
    "reweighing": "Reweighing",
    "exponentiated_gradient": "Exponentiated gradient",
    "threshold_optimizer": "Threshold optimiser (uses group)",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def make_all(metrics: dict, evs: dict, test: pd.DataFrame, spec, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    M = metrics["models"]
    rec = metrics["recommended"]["model"]
    tag = " (SYNTHETIC data)" if metrics["data"]["synthetic"] else ""

    # 1 ROC
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for k, c, lab in [
        ("scorecard_woe_lr", MUTED, "WOE scorecard"),
        ("gbm", VIOLET, "GBM"),
        ("gbm_monotone", BLUE, "GBM monotone"),
        (rec, ORANGE, f"Recommended: {rec}"),
    ]:
        fpr, tpr, _ = roc_curve(test["bad"], evs[k].p_bad_test)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lab} (AUC {M[k]['test']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set_xlabel("False positive rate (goods flagged)")
    ax.set_ylabel("True positive rate (bads caught)")
    ax.set_title("Out-of-time ROC" + tag, fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "roc_scorecard_vs_gbm.png", dpi=130)
    plt.close(fig)

    # 2 AIR before/after with bootstrap CIs
    base = [k for k in M if M[k]["family"] in ("baseline", "challenger", "candidate")]
    order = list(
        dict.fromkeys(
            base
            + [metrics["lda_search"]["selected"]]
            + [k for k in M if M[k]["family"] not in ("baseline", "challenger", "candidate", "lda_candidate")]
        )
    )
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(order) + 1.6))
    for i, k in enumerate(order):
        f = M[k]["fairness"]
        lo, hi = f["air_min_ci"]
        c = FAMILY_COLOR[M[k]["family"]]
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(f["air_min"], i, "o", ms=8, color=c, mec="white", mew=1.5)
    ax.axvline(0.8, color=INK, lw=1, ls="--")
    ax.text(0.8, len(order) - 0.4, " four-fifths line (0.80)", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k + ("  <- recommended" if k == rec else "") for k in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Minimum AIR across protected groups, 95% bootstrap CI")
    ax.set_title("Adverse impact ratio before and after mitigation" + tag, fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "air_before_after.png", dpi=130)
    plt.close(fig)

    # 3 Frontier
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    pts = []
    for fam in ["lda_candidate", "baseline", "challenger", "reweighing", "exponentiated_gradient", "threshold_optimizer", "candidate"]:
        ks = [k for k in M if M[k]["family"] == fam]
        if not ks:
            continue
        x = [M[k]["fairness"]["air_min"] for k in ks]
        y = [M[k]["test"]["balanced_accuracy"] for k in ks]
        pts += list(zip(x, y, ks))
        ax.scatter(x, y, s=90 if fam == "candidate" else 55, color=FAMILY_COLOR[fam], edgecolor="white", lw=1.5, label=FAMILY_LABEL[fam], zorder=3)
    pts.sort(key=lambda p: -p[0])
    front, best = [], -1.0
    for x, y, _k in pts:
        if y > best:
            front.append((x, y))
            best = y
    front.sort()
    ax.plot([p[0] for p in front], [p[1] for p in front], color=MUTED, lw=1.2, ls="-", zorder=2, label="Pareto frontier")
    r = M[rec]
    ax.annotate(
        "recommended", (r["fairness"]["air_min"], r["test"]["balanced_accuracy"]), xytext=(8, -14), textcoords="offset points", fontsize=8, color=INK
    )
    ax.axvline(0.8, color=INK, lw=1, ls="--")
    ax.set_xlabel("Minimum adverse impact ratio (higher = fairer)")
    ax.set_ylabel("Balanced accuracy of approve decision vs recorded outcome")
    ax.set_title("Accuracy-fairness trade-off, out-of-time test" + tag, fontsize=11, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fairness_accuracy_frontier.png", dpi=130)
    plt.close(fig)

    # 4 Approval rates by group
    groups = sorted(M["gbm_monotone"]["fairness"]["approval_rate"])
    show = [("gbm_monotone", BLUE, "Candidate (GBM monotone, with geography)"), (rec, ORANGE, f"Recommended ({rec})")]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    w = 0.38
    for j, (k, c, lab) in enumerate(show):
        vals = [M[k]["fairness"]["approval_rate"][g] for g in groups]
        xs = np.arange(len(groups)) + (j - 0.5) * w
        ax.bar(xs, vals, width=w - 0.03, color=c, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"group {g}" + (" (ref)" if g == spec.reference_group else "") for g in groups])
    ax.set_ylabel("Approval rate")
    ax.set_ylim(0, 1)
    ax.set_title("Approval rate by protected group" + tag, fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "approval_by_group.png", dpi=130)
    plt.close(fig)

    # 5 SHAP importance and top-reason frequency
    E = metrics["explanations"]
    imp = pd.Series(E["mean_abs_shap"]).sort_values()
    freq = pd.Series(E["top_reason_frequency"]).reindex(imp.index).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    axes[0].barh(imp.index, imp.values, color=BLUE)
    axes[0].set_xlabel("Mean |SHAP| (log-odds of bad)")
    axes[0].set_title("Global importance, recommended model", fontsize=10, color=INK)
    axes[1].barh(freq.index, freq.values / E["declined_test_applicants"], color=ORANGE)
    axes[1].set_xlabel("Share of declined applicants")
    axes[1].set_title("Principal (rank-1) adverse-action reason", fontsize=10, color=INK)
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(out / "shap_reason_codes.png", dpi=130)
    plt.close(fig)

    # 6 proxy strength
    P = pd.Series(metrics["proxy_analysis"]["single_feature_auc"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(P.index, P.values - 0.5, left=0.5, color=[ORANGE if f in spec.categorical else BLUE for f in P.index])
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_xlabel("AUC for predicting protected-group membership from one feature (0.5 = none)")
    ax.set_title("Proxy strength of each feature" + tag, fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "proxy_strength.png", dpi=130)
    plt.close(fig)

    # 7 monitoring: feature PSI + monthly AIR (two charts, no dual axis)
    S = metrics["stability"]["feature_psi_train_vs_test"]
    mon = metrics["recommended"]["stability"]["monthly"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ps = pd.Series(S).sort_values()
    axes[0].barh(ps.index, ps.values, color=[ORANGE if v >= 0.10 else BLUE for v in ps.values])
    axes[0].axvline(0.10, color=MUTED, lw=1, ls="--")
    axes[0].axvline(0.25, color=INK, lw=1, ls="--")
    axes[0].set_xlabel("PSI, training vs out-of-time test (0.10 monitor, 0.25 action)")
    axes[0].set_title("Feature stability", fontsize=10, color=INK)
    if mon:
        mm = [r["month"] for r in mon]
        axes[1].plot(mm, [r["air_min_candidate"] for r in mon], "-o", color=BLUE, lw=2, ms=6, label="candidate (GBM monotone)")
        axes[1].plot(mm, [r["air_min_recommended"] for r in mon], "-o", color=ORANGE, lw=2, ms=6, label="recommended")
        axes[1].axhline(0.8, color=INK, lw=1, ls="--")
        axes[1].set_xlabel("Application month (out-of-time)")
        axes[1].set_ylabel("Minimum AIR")
        axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("Monthly fairness monitor", fontsize=10, color=INK)
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(out / "monitoring_psi_air.png", dpi=130)
    plt.close(fig)
