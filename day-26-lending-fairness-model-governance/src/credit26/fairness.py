"""Fairness audit: adverse impact ratio, demographic parity, equalised odds (Fairlearn), bootstrap CIs, proxy analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

AIR_THRESHOLD = 0.80  # four-fifths rule


def fairness_point(approve: np.ndarray, y_good: np.ndarray, groups: np.ndarray, reference: str) -> dict:
    """Point estimates. Favourable outcome = approve; 'positive' truth label = good (did not default)."""
    mf = MetricFrame(
        metrics={"approval_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=y_good,
        y_pred=approve,
        sensitive_features=groups,
    )
    by = mf.by_group
    ref = by.loc[reference, "approval_rate"]
    air = {g: float(by.loc[g, "approval_rate"] / ref) for g in by.index if g != reference}
    return {
        "approval_rate": {g: float(v) for g, v in by["approval_rate"].items()},
        "tpr_good": {g: float(v) for g, v in by["tpr"].items()},
        "fpr_bad": {g: float(v) for g, v in by["fpr"].items()},
        "air": air,
        "air_min": float(min(air.values())),
        "dp_difference": float(demographic_parity_difference(y_good, approve, sensitive_features=groups)),
        "eo_difference": float(equalized_odds_difference(y_good, approve, sensitive_features=groups)),
    }


def _metrics_from_counts(C: np.ndarray, levels: list[str], ref_idx: int) -> dict[str, np.ndarray]:
    """C: (B, G, 2, 2) counts indexed [boot, group, y_good, approve]. Vectorised metrics per bootstrap draw."""
    n_g = C.sum(axis=(2, 3))
    sel = C[:, :, :, 1].sum(axis=2) / np.maximum(n_g, 1)
    tpr = C[:, :, 1, 1] / np.maximum(C[:, :, 1, :].sum(axis=2), 1)
    fpr = C[:, :, 0, 1] / np.maximum(C[:, :, 0, :].sum(axis=2), 1)
    air = sel / np.maximum(sel[:, [ref_idx]], 1e-12)
    others = [i for i in range(len(levels)) if i != ref_idx]
    return {
        "air": air,
        "air_min": air[:, others].min(axis=1),
        "dp_difference": sel.max(axis=1) - sel.min(axis=1),
        "eo_difference": np.maximum(tpr.max(axis=1) - tpr.min(axis=1), fpr.max(axis=1) - fpr.min(axis=1)),
    }


def cell_counts(approve: np.ndarray, y_good: np.ndarray, groups: np.ndarray, levels: list[str]) -> np.ndarray:
    gi = pd.Categorical(groups, categories=levels).codes
    idx = gi * 4 + np.asarray(y_good, dtype=int) * 2 + np.asarray(approve, dtype=int)
    return np.bincount(idx, minlength=len(levels) * 4).reshape(len(levels), 2, 2)


def fairness_bootstrap(approve: np.ndarray, y_good: np.ndarray, groups: np.ndarray, reference: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """Nonparametric bootstrap (applicants resampled with replacement) via multinomial draws on the 4G cell counts."""
    levels = sorted(set(groups))
    ref_idx = levels.index(reference)
    C0 = cell_counts(approve, y_good, groups, levels)
    n = int(C0.sum())
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, (C0 / n).ravel(), size=n_boot).reshape(n_boot, len(levels), 2, 2)
    m = _metrics_from_counts(draws, levels, ref_idx)
    ci = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
    out = {
        "n_boot": n_boot,
        "air_ci": {g: ci(m["air"][:, i]) for i, g in enumerate(levels) if g != reference},
        "air_min_ci": ci(m["air_min"]),
        "dp_difference_ci": ci(m["dp_difference"]),
        "eo_difference_ci": ci(m["eo_difference"]),
        "p_air_min_below_0_8": float(np.mean(m["air_min"] < AIR_THRESHOLD)),
    }
    return out


def audit(approve: np.ndarray, y_good: np.ndarray, groups: np.ndarray, reference: str, n_boot: int = 1000, seed: int = 0) -> dict:
    pt = fairness_point(approve, y_good, groups, reference)
    pt.update(fairness_bootstrap(approve, y_good, groups, reference, n_boot=n_boot, seed=seed))
    pt["passes_four_fifths"] = bool(pt["air_min"] >= AIR_THRESHOLD)
    return pt


def _oof_target_encode(x: pd.Series, t: np.ndarray, seed: int = 0) -> np.ndarray:
    out = np.zeros(len(x))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    xs = x.astype(str).to_numpy()
    for tr, te in skf.split(xs, t):
        m = pd.Series(t[tr]).groupby(xs[tr]).mean()
        out[te] = pd.Series(xs[te]).map(m).fillna(t[tr].mean()).to_numpy()
    return out


def proxy_strength(df: pd.DataFrame, features: list[str], categorical: list[str], protected: str, reference: str, seed: int = 0) -> dict[str, float]:
    """Single-feature AUC for predicting membership of a protected (non-reference) group; 0.5 = no information."""
    t = (df[protected].astype(str) != reference).to_numpy().astype(int)
    out = {}
    for f in features:
        s = _oof_target_encode(df[f], t, seed) if f in categorical else df[f].to_numpy(dtype=float)
        a = roc_auc_score(t, s)
        out[f] = float(max(a, 1 - a))
    return out


def joint_proxy_auc(df: pd.DataFrame, features: list[str], categorical: list[str], protected: str, reference: str, seed: int = 0) -> float:
    """5-fold CV AUC of a LightGBM classifier predicting protected-group membership from the model's features."""
    import lightgbm as lgb

    from .models import frame

    t = (df[protected].astype(str) != reference).to_numpy().astype(int)
    X = frame(df, features, [c for c in features if c in categorical])
    p = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, t):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=1, verbose=-1)
        m.fit(X.iloc[tr], t[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(t, p))
