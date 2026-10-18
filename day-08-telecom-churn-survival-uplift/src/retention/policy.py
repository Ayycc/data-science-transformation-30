"""Budgeted targeting policies and their evaluation against ground truth and the randomised data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def greedy_budget(priority: np.ndarray, cost: np.ndarray, budget: float, eligible: np.ndarray | None = None) -> np.ndarray:
    """Target customers in descending ``priority`` until the (estimated) spend reaches ``budget``."""
    eligible = np.ones(len(priority), dtype=bool) if eligible is None else eligible
    idx = np.where(eligible)[0]
    idx = idx[np.argsort(-priority[idx], kind="stable")]
    take = idx[np.cumsum(cost[idx]) <= budget]
    mask = np.zeros(len(priority), dtype=bool)
    mask[take] = True
    return mask


def value_policy(net_hat: np.ndarray, cost_hat: np.ndarray, budget: float) -> np.ndarray:
    """Knapsack-greedy: only positive expected net value, ranked by expected net value per $ spent."""
    return greedy_budget(net_hat / cost_hat, cost_hat, budget, eligible=net_hat > 0)


def rct_incremental_retained(mask: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Difference-in-means estimate (and SE) of extra customers retained if the targeted set is treated."""
    yy, tt = y[mask], t[mask]
    if mask.sum() == 0 or tt.min() == tt.max():
        return 0.0, 0.0
    a, b = yy[tt == 1], yy[tt == 0]
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(diff * mask.sum()), float(se * mask.sum())


def evaluate_policy(name: str, mask: np.ndarray, hold: pd.DataFrame, net_hat: np.ndarray | None, cost_hat: np.ndarray) -> dict:
    """Score a targeting set on the holdout.

    True values come from the known data-generating process (``dv_true``, ``cost_true``).
    ``dr_net`` is the doubly robust estimate an analyst could compute without ground truth.
    """
    spend = float(hold["cost_true"].to_numpy()[mask].sum())
    inc = float(hold["dv_true"].to_numpy()[mask].sum())
    rct, rct_se = rct_incremental_retained(mask, hold["retained"].to_numpy(), hold["treatment"].to_numpy())
    out = {
        "policy": name,
        "n_targeted": int(mask.sum()),
        "share_targeted": float(mask.mean()),
        "spend_true": spend,
        "incremental_revenue_true": inc,
        "net_incremental_revenue_true": inc - spend,
        "roi_true": (inc - spend) / spend if spend > 0 else 0.0,
        "true_incremental_retained": float(hold["tau_true"].to_numpy()[mask].sum()),
        "rct_incremental_retained": rct,
        "rct_incremental_retained_se": rct_se,
        "sleeping_dogs_targeted": int((hold["tau_true"].to_numpy()[mask] < 0).sum()),
        "dr_net_estimate": float((hold["dr_value"].to_numpy()[mask] - cost_hat[mask]).sum()),
        "spend_estimated": float(cost_hat[mask].sum()),
    }
    if net_hat is not None:
        es = float(cost_hat[mask].sum())
        out["expected_net"] = float(net_hat[mask].sum())
        out["expected_roi"] = float(net_hat[mask].sum() / es) if es > 0 else 0.0
    return out


def fit_policy_tree(X: pd.DataFrame, pseudo_net: np.ndarray, depth: int, seed: int) -> DecisionTreeClassifier:
    """Policy learning with doubly robust scores: a shallow tree that classifies sign(net) weighted by |net|."""
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=500, random_state=seed)
    tree.fit(X, (pseudo_net > 0).astype(int), sample_weight=np.abs(pseudo_net))
    return tree


def leaf_values(tree: DecisionTreeClassifier, X_fit: pd.DataFrame, pseudo_net: np.ndarray, X_new: pd.DataFrame) -> np.ndarray:
    """Average DR net value of the training leaf each new row lands in."""
    leaf_fit = tree.apply(X_fit)
    means = pd.Series(pseudo_net).groupby(leaf_fit).mean()
    return means.reindex(tree.apply(X_new)).fillna(0.0).to_numpy()
