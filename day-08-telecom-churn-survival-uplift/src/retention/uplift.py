"""Uplift models (T-learner, X-learner, uplift random forest) and Qini / uplift curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseTClassifier, BaseXClassifier
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import FEATURES

UPLIFT_FEATURES = FEATURES + ["tenure_at_campaign"]


def _clf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, random_state=seed
    )


def _reg(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, random_state=seed
    )


def uplift_frame(campaign: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out["tenure_at_campaign"] = campaign["tenure_at_campaign"].to_numpy().astype(float)
    out["treatment"] = campaign["treatment"].to_numpy().astype(int)
    out["retained"] = 1 - campaign["churn_6m"].to_numpy().astype(int)  # outcome: stayed through the window
    return out


def fit_predict_all(train: pd.DataFrame, score: pd.DataFrame, cfg: dict, seed: int) -> dict[str, np.ndarray]:
    """Fit the three uplift learners on ``train``; return predicted retention uplift for ``score``.

    Positive uplift = the offer raises the probability of staying 6 more months.
    """
    Xtr, t, y = train[UPLIFT_FEATURES].to_numpy(), train["treatment"].to_numpy(), train["retained"].to_numpy()
    Xs = score[UPLIFT_FEATURES].to_numpy()
    p_tr = np.full(len(train), t.mean())
    p_s = np.full(len(score), t.mean())
    out: dict[str, np.ndarray] = {}

    tl = BaseTClassifier(learner=_clf(seed), control_name=0)
    tl.fit(Xtr, treatment=t, y=y)
    out["t_learner"] = np.asarray(tl.predict(Xs, verbose=False)).ravel()

    xl = BaseXClassifier(outcome_learner=_clf(seed), effect_learner=_reg(seed), control_name=0)
    xl.fit(Xtr, treatment=t, y=y, p=p_tr)
    out["x_learner"] = np.asarray(xl.predict(Xs, p=p_s, verbose=False)).ravel()

    rng = np.random.default_rng(seed)
    n = min(cfg["uplift_rf_train_n"], len(train))
    idx = rng.choice(len(train), n, replace=False)
    urf = UpliftRandomForestClassifier(
        control_name="control",
        n_estimators=cfg["uplift_rf_trees"],
        max_depth=cfg["uplift_rf_max_depth"],
        min_samples_leaf=cfg["uplift_rf_min_leaf"],
        min_samples_treatment=50,
        max_features=6,
        evaluationFunction="KL",
        random_state=seed,
        n_jobs=4,
    )
    tr_labels = np.where(t[idx] == 1, "treatment", "control")
    urf.fit(Xtr[idx], treatment=tr_labels, y=y[idx])
    out["uplift_rf"] = np.asarray(urf.predict(Xs)).ravel()
    return out


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: incremental responders when targeting the top fraction by ``score``.

    Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k), computed on observed randomised outcomes.
    Returns (fraction targeted, Q as a fraction of the population).
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    yt = np.cumsum(y * t)
    yc = np.cumsum(y * (1 - t))
    nt = np.cumsum(t)
    nc = np.cumsum(1 - t)
    q = yt - yc * np.divide(nt, nc, out=np.zeros_like(nt, dtype=float), where=nc > 0)
    n = len(y)
    ks = np.linspace(0, n, n_points).astype(int)
    qq = np.concatenate([[0.0], q])[ks]
    return ks / n, qq / n


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area between the Qini curve and the random-targeting diagonal (per customer)."""
    f, q = qini_curve(y, t, score)
    rand = f * q[-1]
    return float(np.trapezoid(q - rand, f))


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Observed uplift (treated minus control retention rate) inside the top fraction."""
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    fr = np.linspace(0.05, 1.0, n_points - 1)
    out = []
    for f in fr:
        k = int(round(f * len(y)))
        yy, tt = y[:k], t[:k]
        out.append(yy[tt == 1].mean() - yy[tt == 0].mean())
    return fr, np.array(out)


def true_gain_curve(tau_true: np.ndarray, score: np.ndarray, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative TRUE expected incremental retained customers (per customer) by targeting depth."""
    order = np.argsort(-score, kind="stable")
    g = np.concatenate([[0.0], np.cumsum(tau_true[order])])
    ks = np.linspace(0, len(score), n_points).astype(int)
    return ks / len(score), g[ks] / len(score)


def aipw_scores(y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: float) -> np.ndarray:
    """Doubly robust (AIPW) pseudo-outcome for the individual retention effect."""
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


def crossfit_outcome_models(df: pd.DataFrame, seed: int, folds: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted mu0(x), mu1(x) for AIPW scores on ``df``."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(df))
    X, t, y = df[UPLIFT_FEATURES].to_numpy(), df["treatment"].to_numpy(), df["retained"].to_numpy()
    mu0, mu1 = np.zeros(len(df)), np.zeros(len(df))
    for k in range(folds):
        tr, te = fold != k, fold == k
        for arm, mu in ((0, mu0), (1, mu1)):
            m = _clf(seed + k).fit(X[tr & (t == arm)], y[tr & (t == arm)])
            mu[te] = m.predict_proba(X[te])[:, 1]
    return mu0, mu1


def fit_outcome_models(df: pd.DataFrame, seed: int):
    X, t, y = df[UPLIFT_FEATURES].to_numpy(), df["treatment"].to_numpy(), df["retained"].to_numpy()
    return [_clf(seed).fit(X[t == arm], y[t == arm]) for arm in (0, 1)]
