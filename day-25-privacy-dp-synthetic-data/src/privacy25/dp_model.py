"""DP logistic regression by objective perturbation (Chaudhuri, Monteleoni & Sarwate 2011, Algorithm 2).

diffprivlib's ``LogisticRegression`` implements the same algorithm, and it is run here as a cross-check.
With diffprivlib 0.6.5 and scikit-learn 1.6 the perturbation vector is added to the *averaged* loss without
the 1/n factor, so the noise is n times larger than the algorithm needs (still private, but at chance level
even at epsilon = 10). ``ObjectivePerturbationLR`` is a small, tested implementation with the 1/n scaling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from diffprivlib.models import LogisticRegression as DPLibLogisticRegression
from scipy import optimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import RARE_GROUP
from .encoding import linear_features


@dataclass
class ObjectivePerturbationLR:
    """epsilon-DP L2-regularised logistic regression; rows must satisfy ||x||_2 <= data_norm (public bound).

    Minimises (1/n) sum log(1 + exp(-y w.z)) + (lam + Delta)/2 ||w||^2 + (1/n) b.w with z = [x / data_norm, 1] / sqrt(2)
    (so ||z|| <= 1), ||b|| ~ Gamma(d, 2/eps') and a uniform direction. eps' and Delta follow Algorithm 2 with c = 1/4.
    """

    epsilon: float
    data_norm: float
    lam: float = 1e-4
    random_state: int = 0

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = X / self.data_norm
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1.0)  # clip, in case a row breaks the public bound
        return np.hstack([X, np.ones((len(X), 1))]) / np.sqrt(2)

    def privacy_terms(self, n: int) -> tuple[float, float]:
        c = 0.25
        eps_p = self.epsilon - np.log(1 + 2 * c / (n * self.lam) + c**2 / (n * self.lam) ** 2)
        delta = 0.0
        if eps_p <= 0:
            delta = c / (n * (np.exp(self.epsilon / 4) - 1)) - self.lam
            eps_p = self.epsilon / 2
        return float(eps_p), float(delta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ObjectivePerturbationLR:
        rng = np.random.default_rng(self.random_state)
        Z = self._z(X)
        n, d = Z.shape
        s = 2 * np.asarray(y) - 1
        eps_p, delta = self.privacy_terms(n)
        direction = rng.normal(size=d)
        b = direction / np.linalg.norm(direction) * rng.gamma(d, 2 / eps_p)
        reg = self.lam + delta

        def f(w: np.ndarray) -> tuple[float, np.ndarray]:
            m = s * (Z @ w)
            loss = np.logaddexp(0, -m).mean()
            g = -(Z * (s / (1 + np.exp(m)))[:, None]).mean(0)
            return loss + reg / 2 * w @ w + b @ w / n, g + reg * w + b / n

        self.w_ = optimize.minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 1000}).x
        self.noise_norm_ = float(np.linalg.norm(b))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1 / (1 + np.exp(-(self._z(X) @ self.w_)))
        return np.column_stack([1 - p, p])


def _scores(model, X: np.ndarray, y: np.ndarray, rare: np.ndarray) -> dict:
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy_rare": float(accuracy_score(y[rare], pred[rare])),
        "auc_rare": float(roc_auc_score(y[rare], p[rare])) if len(np.unique(y[rare])) == 2 else float("nan"),
        "accuracy_other": float(accuracy_score(y[~rare], pred[~rare])),
        "auc_other": float(roc_auc_score(y[~rare], p[~rare])),
    }


def loss_threshold_mia_auc(model, X_in: np.ndarray, y_in: np.ndarray, X_out: np.ndarray, y_out: np.ndarray) -> float:
    """Yeom et al. loss-threshold membership attack (members have lower loss). Returns the attack AUC."""

    def loss(X, y):
        p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    s = np.concatenate([-loss(X_in, y_in), -loss(X_out, y_out)])
    lab = np.concatenate([np.ones(len(y_in)), np.zeros(len(y_out))])
    return float(roc_auc_score(lab, s))


def choose_lambda(pilot_train: pd.DataFrame, pilot_test: pd.DataFrame, epsilon: float, grid: list[float], seeds: int) -> float:
    """Pick the regulariser on a *pilot* dataset that is not the confidential file (so it costs no budget)."""
    X, y, dn = linear_features(pilot_train)
    Xt, yt, _ = linear_features(pilot_test)
    best, best_auc = grid[0], -1.0
    for lam in grid:
        auc = np.mean([roc_auc_score(yt, ObjectivePerturbationLR(epsilon, dn, lam, 500 + s).fit(X, y).predict_proba(Xt)[:, 1]) for s in range(seeds)])
        if auc > best_auc:
            best, best_auc = lam, float(auc)
    return best


def run_dp_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pilot: tuple[pd.DataFrame, pd.DataFrame],
    epsilons: list[float],
    seeds: int,
    seed: int,
    lam_grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2),
    diffprivlib_epsilons: tuple[float, ...] = (1.0, 10.0),
) -> dict:
    Xtr, ytr, data_norm = linear_features(train)
    Xte, yte, _ = linear_features(test)
    rare = (test[RARE_GROUP[0]] == RARE_GROUP[1]).to_numpy()
    rng = np.random.default_rng(seed)
    m_in = rng.choice(len(ytr), size=min(3000, len(ytr), len(yte)), replace=False)
    m_out = rng.choice(len(yte), size=len(m_in), replace=False)

    def mia(m) -> float:
        return loss_threshold_mia_auc(m, Xtr[m_in], ytr[m_in], Xte[m_out], yte[m_out])

    base = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr)
    res: dict = {
        "data_norm": data_norm,
        "n_features": int(Xtr.shape[1]),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_test_rare": int(rare.sum()),
        "majority_class_accuracy": float(max(yte.mean(), 1 - yte.mean())),
        "majority_class_accuracy_rare": float(max(yte[rare].mean(), 1 - yte[rare].mean())),
        "baseline": _scores(base, Xte, yte, rare) | {"mia_auc": mia(base)},
        "dp": [],
        "diffprivlib_check": [],
    }
    for eps in epsilons:
        lam = choose_lambda(pilot[0], pilot[1], eps, list(lam_grid), seeds=3)
        runs = []
        for s in range(seeds):
            m = ObjectivePerturbationLR(eps, data_norm, lam, seed + 1000 * s + 7).fit(Xtr, ytr)
            runs.append(_scores(m, Xte, yte, rare) | {"mia_auc": mia(m)})
        agg = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}
        agg |= {f"{k}_sd": float(np.nanstd([r[k] for r in runs])) for k in ("accuracy", "auc", "accuracy_rare", "auc_rare")}
        res["dp"].append({"epsilon": eps, "lambda": lam, "seeds": seeds, **agg})
    for eps in diffprivlib_epsilons:
        runs = []
        for s in range(seeds):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = DPLibLogisticRegression(epsilon=eps, data_norm=data_norm, C=1.0, max_iter=2000, random_state=seed + s).fit(Xtr, ytr)
            runs.append(_scores(m, Xte, yte, rare))
        res["diffprivlib_check"].append({"epsilon": eps, **{k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}})
    return res
