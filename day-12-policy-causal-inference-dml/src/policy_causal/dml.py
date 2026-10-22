"""Double machine learning: ATE (LinearDML, DR-learner) and heterogeneous effects (causal forest)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from econml.dr import LinearDRLearner
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def nuisance_models(seed: int, quick: bool = False) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingClassifier]:
    it = 80 if quick else 250
    reg = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, random_state=seed)
    return reg, clf


def _ci(point: float, lo: float, hi: float) -> dict[str, float]:
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "se": float((hi - lo) / (2 * 1.959964))}


def linear_dml_ate(
    df: pd.DataFrame, outcome: str, modifiers: list[str], controls: list[str], folds: int, seed: int, quick: bool = False
) -> dict[str, float]:
    """Partially linear DML with a CATE that is linear in ``modifiers``; ATE = average CATE over the sample.

    Note: LinearDML with *no* modifiers estimates an overlap-weighted effect, not the ATE, when effects
    are heterogeneous - so the modifiers are needed for an ATE.
    """
    reg, clf = nuisance_models(seed, quick)
    m = LinearDML(model_y=reg, model_t=clf, discrete_treatment=True, cv=folds, random_state=seed)
    x = df[modifiers].to_numpy(dtype=float)
    m.fit(df[outcome].to_numpy(dtype=float), df["treated"].to_numpy(), X=x, W=df[controls].to_numpy(dtype=float))
    lo, hi = m.ate_interval(x, alpha=0.05)
    out = _ci(m.ate(x), lo, hi)
    out["coef"] = {k: float(v) for k, v in zip(modifiers, np.ravel(m.coef_), strict=True)}
    out["intercept"] = float(np.ravel(m.intercept_)[0])
    return out


def overlap_weighted_dml(
    df: pd.DataFrame, outcome: str, confounders: list[str], folds: int, seed: int, quick: bool = False
) -> dict[str, float]:
    """LinearDML with a constant effect: the classic Chernozhukov et al. partialling-out estimator."""
    reg, clf = nuisance_models(seed, quick)
    m = LinearDML(model_y=reg, model_t=clf, discrete_treatment=True, cv=folds, random_state=seed)
    m.fit(df[outcome].to_numpy(dtype=float), df["treated"].to_numpy(), X=None, W=df[confounders].to_numpy(dtype=float))
    lo, hi = m.ate_interval(alpha=0.05)
    return _ci(m.ate(), lo, hi)


def dr_learner_ate(df: pd.DataFrame, outcome: str, confounders: list[str], folds: int, seed: int, quick: bool = False) -> dict[str, float]:
    """Cross-fitted AIPW (doubly robust) ATE via econml's LinearDRLearner with no effect modifiers."""
    reg, clf = nuisance_models(seed, quick)
    m = LinearDRLearner(model_regression=reg, model_propensity=clf, cv=folds, random_state=seed, min_propensity=0.01)
    m.fit(df[outcome].to_numpy(dtype=float), df["treated"].to_numpy(), X=None, W=df[confounders].to_numpy(dtype=float))
    lo, hi = m.ate_interval(alpha=0.05)
    return _ci(m.ate(), lo, hi)


def causal_forest(
    df: pd.DataFrame, outcome: str, modifiers: list[str], controls: list[str], folds: int, trees: int, seed: int, quick: bool = False
) -> tuple[CausalForestDML, dict[str, object], np.ndarray]:
    reg, clf = nuisance_models(seed, quick)
    cf = CausalForestDML(
        model_y=reg,
        model_t=clf,
        discrete_treatment=True,
        cv=folds,
        n_estimators=trees,
        min_samples_leaf=20,
        max_samples=0.45,
        honest=True,
        inference=True,
        random_state=seed,
        n_jobs=-1,
    )
    x = df[modifiers].to_numpy(dtype=float)
    cf.fit(df[outcome].to_numpy(dtype=float), df["treated"].to_numpy(), X=x, W=df[controls].to_numpy(dtype=float) if controls else None)
    cate = cf.effect(x)
    ate = float(np.ravel(cf.ate_)[0])
    se = float(np.ravel(cf.ate_stderr_)[0])
    summary = {
        "ate_dr": _ci(ate, ate - 1.959964 * se, ate + 1.959964 * se),
        "feature_importance": {k: float(v) for k, v in zip(modifiers, cf.feature_importances_, strict=True)},
    }
    return cf, summary, cate


def subgroup_table(
    cf: CausalForestDML, df: pd.DataFrame, modifiers: list[str], cate: np.ndarray, true_tau: np.ndarray | None
) -> pd.DataFrame:
    """Average CATE with 95% CI in policy-relevant subgroups, next to the true subgroup effect."""
    groups = {
        "all": np.ones(len(df), dtype=bool),
        "unemployed at baseline": df["unemployed_base"].to_numpy() == 1,
        "employed at baseline": df["unemployed_base"].to_numpy() == 0,
        "age < 30": df["age"].to_numpy() < 30,
        "age 30-44": (df["age"].to_numpy() >= 30) & (df["age"].to_numpy() < 45),
        "age 45+": df["age"].to_numpy() >= 45,
        "educ <= 11 (no diploma)": df["educ"].to_numpy() <= 11,
        "educ 12-13": (df["educ"].to_numpy() >= 12) & (df["educ"].to_numpy() <= 13),
        "educ >= 14": df["educ"].to_numpy() >= 14,
        "women": df["female"].to_numpy() == 1,
        "men": df["female"].to_numpy() == 0,
    }
    x = df[modifiers].to_numpy(dtype=float)
    rows = []
    for name, mask in groups.items():
        if mask.sum() < 30:
            continue
        lo, hi = cf.ate_interval(x[mask], alpha=0.05)
        row = {"group": name, "n": int(mask.sum()), "cate_mean": float(cate[mask].mean()), "ci_low": float(lo), "ci_high": float(hi)}
        if true_tau is not None:
            row["true_effect"] = float(true_tau[mask].mean())
            row["truth_in_ci"] = bool(lo <= row["true_effect"] <= hi)
        rows.append(row)
    return pd.DataFrame(rows)


def targeting_gain(cate: np.ndarray, true_tau: np.ndarray, share: float = 0.3) -> dict[str, float]:
    """True average effect among the top ``share`` ranked by estimated CATE vs everyone (random targeting)."""
    k = int(round(share * len(cate)))
    top = np.argsort(-cate)[:k]
    oracle = np.argsort(-true_tau)[:k]
    return {
        "share_targeted": share,
        "true_effect_top_by_cate": float(true_tau[top].mean()),
        "true_effect_random": float(true_tau.mean()),
        "true_effect_oracle_top": float(true_tau[oracle].mean()),
        "cate_true_corr": float(np.corrcoef(cate, true_tau)[0, 1]),
    }


def coverage_study(
    n_people: int, seeds: list[int], modifiers: list[str], controls: list[str], folds: int, quick: bool = False
) -> dict[str, object]:
    """Re-simulate with fresh seeds and record how often the headline LinearDML 95% CI covers the true SATE."""
    from .simulate import observed, simulate_individuals, true_effects

    rows = []
    for s in seeds:
        full = simulate_individuals(n_people, s)
        df = observed(full)
        tr = true_effects(full)
        for y, key in (("earnings_post", "ate_earnings"), ("employed_post", "ate_employment")):
            r = linear_dml_ate(df, y, modifiers, controls, folds, s, quick)
            rows.append({"seed": s, "outcome": y, "truth": tr[key], **{k: r[k] for k in ("estimate", "ci_low", "ci_high")}})
    tab = pd.DataFrame(rows)
    tab["covered"] = (tab["ci_low"] <= tab["truth"]) & (tab["truth"] <= tab["ci_high"])
    tab["bias"] = tab["estimate"] - tab["truth"]
    summ = {}
    for y, g in tab.groupby("outcome"):
        summ[y] = {
            "replications": len(g),
            "coverage": float(g["covered"].mean()),
            "mean_bias": float(g["bias"].mean()),
            "mean_abs_bias": float(g["bias"].abs().mean()),
            "mean_ci_width": float((g["ci_high"] - g["ci_low"]).mean()),
        }
    return {"summary": summ, "table": tab}
