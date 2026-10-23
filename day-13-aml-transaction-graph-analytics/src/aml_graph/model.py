"""Node classification: tabular-only vs graph-feature LightGBM models, grouped split, bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .config import ModelConfig


@dataclass
class FitResult:
    name: str
    columns: list[str]
    model: lgb.LGBMClassifier
    test_score: np.ndarray


def grouped_split(acc: pd.DataFrame, cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Accounts of one laundering scheme never straddle train and test."""
    inst = acc["instance_id"].to_numpy()
    groups = np.where(inst >= 0, inst, acc["account_id"].to_numpy() + 10_000_000)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(gss.split(acc, groups=groups))
    return np.sort(tr), np.sort(te)


def fit_model(name: str, x: pd.DataFrame, y: np.ndarray, tr: np.ndarray, te: np.ndarray, cfg: ModelConfig) -> FitResult:
    m = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=cfg.seed,
        n_jobs=4,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    m.fit(x.iloc[tr], y[tr])
    return FitResult(name, list(x.columns), m, m.predict_proba(x.iloc[te])[:, 1])


def operating_point(y: np.ndarray, s: np.ndarray, target_recall: float) -> dict:
    p, r, t = precision_recall_curve(y, s)
    ok = np.where(r[:-1] >= target_recall)[0]
    i = ok[-1]  # highest threshold that still reaches the target recall
    thr = t[i]
    alerts = int((s >= thr).sum())
    tp = int(((s >= thr) & (y == 1)).sum())
    return {
        "threshold": float(thr),
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": alerts - tp,
        "precision": tp / alerts,
        "recall": tp / int(y.sum()),
    }


def evaluate(y: np.ndarray, s: np.ndarray, cfg: ModelConfig) -> dict:
    k = max(1, int(round(0.01 * len(y))))
    top = np.argsort(-s)[:k]
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision_at_top1pct": float(y[top].mean()),
        "recall_at_top1pct": float(y[top].sum() / y.sum()),
        f"at_recall_{int(cfg.target_recall * 100)}": operating_point(y, s, cfg.target_recall),
    }


def bootstrap_pr_auc_diff(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, n: int, seed: int) -> dict:
    """Paired bootstrap of PR-AUC(a) - PR-AUC(b) over test accounts."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        diffs.append(average_precision_score(y[i], s_a[i]) - average_precision_score(y[i], s_b[i]))
    d = np.array(diffs)
    return {
        "mean": float(d.mean()),
        "ci95_lo": float(np.quantile(d, 0.025)),
        "ci95_hi": float(np.quantile(d, 0.975)),
        "share_positive": float((d > 0).mean()),
        "n": int(len(d)),
    }


def per_typology_recall(acc_test: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    out = {}
    for typ, d in acc_test[acc_test["is_laundering"]].groupby("typology"):
        out[typ] = float((s[d.index.to_numpy()] >= thr).mean())
    return out


def shap_contributions(fit: FitResult, x: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions from LightGBM (last column is the bias)."""
    c = fit.model.booster_.predict(x[fit.columns], pred_contrib=True)
    return pd.DataFrame(c[:, :-1], columns=fit.columns, index=x.index)
