"""Metrics and cost-sensitive threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def outcome_costs(amount: np.ndarray, cost: dict) -> dict[str, np.ndarray]:
    """Per-transaction cost of each outcome under the cost matrix (fixed + pct * amount)."""
    amount = np.asarray(amount, dtype=np.float64)
    return {
        k: cost[k]["fixed"] + cost[k]["pct_of_amount"] * amount
        for k in ("true_positive", "false_positive", "false_negative", "true_negative")
    }


def cost_curve(y: np.ndarray, score: np.ndarray, amount: np.ndarray, cost: dict) -> dict[str, np.ndarray]:
    """Total cost for every distinct threshold (alert when score >= threshold).

    Returns arrays ordered by decreasing threshold, starting with "alert nothing" (threshold = +inf).
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    c = outcome_costs(amount, cost)
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    # moving one transaction from "approve" to "alert" changes the cost by delta
    delta = np.where(yy, c["true_positive"][order] - c["false_negative"][order],
                     c["false_positive"][order] - c["true_negative"][order])  # fmt: skip
    base = float(np.where(y, c["false_negative"], c["true_negative"]).sum())
    cum = np.concatenate([[0.0], np.cumsum(delta)])
    # only cut between distinct scores
    last_of_run = np.concatenate([np.flatnonzero(s[1:] != s[:-1]), [len(s) - 1]])
    k = np.concatenate([[0], last_of_run + 1])  # number of alerts
    thr = np.concatenate([[np.inf], s[last_of_run]])
    tp = np.concatenate([[0], np.cumsum(yy)])[k]
    fp = k - tp
    return {"threshold": thr, "n_alerts": k, "tp": tp, "fp": fp, "total_cost": base + cum[k]}


def select_threshold(y: np.ndarray, score: np.ndarray, amount: np.ndarray, cost: dict) -> dict:
    """Pick the threshold that minimises total expected cost on (validation) data."""
    cc = cost_curve(y, score, amount, cost)
    i = int(np.argmin(cc["total_cost"]))
    return {
        "threshold": float(cc["threshold"][i]),
        "total_cost": float(cc["total_cost"][i]),
        "n_alerts": int(cc["n_alerts"][i]),
        "curve": cc,
    }


def apply_threshold(y: np.ndarray, score: np.ndarray, amount: np.ndarray, cost: dict, threshold: float) -> dict:
    """Confusion matrix and dollar breakdown at a fixed threshold."""
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype=np.float64)
    alert = np.asarray(score) >= threshold
    c = outcome_costs(amount, cost)
    tp, fp, fn, tn = alert & y, alert & ~y, ~alert & y, ~alert & ~y
    parts = {
        "true_positive_cost": float(c["true_positive"][tp].sum()),
        "false_positive_cost": float(c["false_positive"][fp].sum()),
        "false_negative_cost": float(c["false_negative"][fn].sum()),
        "true_negative_cost": float(c["true_negative"][tn].sum()),
    }
    n_neg = int((~y).sum())
    return {
        "threshold": float(threshold),
        "alerts": int(alert.sum()),
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "tn": int(tn.sum()),
        "recall": float(tp.sum() / max(1, y.sum())),
        "precision": float(tp.sum() / max(1, alert.sum())),
        "fpr": float(fp.sum() / max(1, n_neg)),
        "fraud_usd_total": float(amount[y].sum()),
        "fraud_usd_caught": float(amount[tp].sum()),
        "fraud_usd_recall": float(amount[tp].sum() / max(1e-9, amount[y].sum())),
        "declined_good_usd": float(amount[fp].sum()),
        **parts,
        "total_cost": float(sum(parts.values())),
    }


def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr_target: float) -> float:
    """Highest recall achievable with false-positive rate <= fpr_target."""
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def ranking_metrics(y: np.ndarray, score: np.ndarray, fprs: tuple[float, ...] = (0.001, 0.005, 0.01)) -> dict:
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    for f in fprs:
        out[f"recall_at_fpr_{f * 100:g}pct"] = recall_at_fpr(y, score, f)
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-mass (quantile) bins, suited to heavily imbalanced scores."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))
