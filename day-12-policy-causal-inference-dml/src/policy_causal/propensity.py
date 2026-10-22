"""Propensity-score methods: IPW (Hajek), 1:1 nearest-neighbour matching, and balance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CLIP = (0.01, 0.99)


def fit_propensity(x: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """Logistic propensity model with all pairwise interactions and squares (analyst does not know the true form)."""
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000),
    )
    model.fit(x.to_numpy(dtype=float), t)
    return np.clip(model.predict_proba(x.to_numpy(dtype=float))[:, 1], *CLIP)


def ipw_weights(t: np.ndarray, e: np.ndarray, estimand: str = "ate") -> np.ndarray:
    if estimand == "ate":
        return np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    if estimand == "att":
        return np.where(t == 1, 1.0, e / (1.0 - e))
    raise ValueError(estimand)


def ipw_estimate(y: np.ndarray, t: np.ndarray, e: np.ndarray, estimand: str = "ate") -> float:
    """Normalised (Hajek) inverse-probability-weighted difference in means."""
    w = ipw_weights(t, e, estimand)
    m1 = np.sum(w * t * y) / np.sum(w * t)
    m0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
    return float(m1 - m0)


def ipw_mean_y0(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    """IPW estimate of E[Y(0)] (used as the baseline for the employment E-value)."""
    w = (1 - t) / (1 - e)
    return float(np.sum(w * y) / np.sum(w))


def match_pairs(t: np.ndarray, e: np.ndarray, caliper_sd: float = 0.2) -> dict[str, np.ndarray]:
    """1:1 nearest-neighbour matching on the logit propensity score, with replacement, both directions.

    Returns, for every unit, the index of its nearest opposite-arm unit and whether it is inside the caliper.
    """
    lp = np.log(e / (1 - e))
    caliper = caliper_sd * lp.std()
    idx_t = np.flatnonzero(t == 1)
    idx_c = np.flatnonzero(t == 0)
    match = np.full(len(t), -1)
    dist = np.full(len(t), np.inf)
    nn_c = NearestNeighbors(n_neighbors=1).fit(lp[idx_c, None])
    d, j = nn_c.kneighbors(lp[idx_t, None])
    match[idx_t], dist[idx_t] = idx_c[j[:, 0]], d[:, 0]
    nn_t = NearestNeighbors(n_neighbors=1).fit(lp[idx_t, None])
    d, j = nn_t.kneighbors(lp[idx_c, None])
    match[idx_c], dist[idx_c] = idx_t[j[:, 0]], d[:, 0]
    return {"match": match, "within": dist <= caliper, "caliper": np.array(caliper)}


def matching_estimate(y: np.ndarray, t: np.ndarray, pairs: dict[str, np.ndarray], estimand: str = "ate") -> float:
    ok = pairs["within"]
    m = pairs["match"]
    y1 = np.where(t == 1, y, y[m])
    y0 = np.where(t == 1, y[m], y)
    keep = ok if estimand == "ate" else ok & (t == 1)
    return float(np.mean(y1[keep] - y0[keep]))


def matched_frequency_weights(t: np.ndarray, pairs: dict[str, np.ndarray]) -> np.ndarray:
    """Frequency weights describing the matched sample used for the ATE (each unit + each time it is reused)."""
    w = pairs["within"].astype(float)
    np.add.at(w, pairs["match"][pairs["within"]], 1.0)
    return w


def smd(x: pd.DataFrame, t: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    """Standardised mean differences (pooled unweighted SD in the denominator, as in Austin 2009)."""
    w = np.ones(len(t)) if w is None else w
    out = {}
    for c in x.columns:
        v = x[c].to_numpy(dtype=float)
        m1 = np.average(v[t == 1], weights=w[t == 1])
        m0 = np.average(v[t == 0], weights=w[t == 0])
        sd = np.sqrt((v[t == 1].var(ddof=1) + v[t == 0].var(ddof=1)) / 2)
        out[c] = (m1 - m0) / sd if sd > 0 else 0.0
    return pd.Series(out)


def variance_ratio(x: pd.DataFrame, t: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    w = np.ones(len(t)) if w is None else w
    out = {}
    for c in x.columns:
        v = x[c].to_numpy(dtype=float)

        def wvar(a: np.ndarray, ww: np.ndarray) -> float:
            mu = np.average(a, weights=ww)
            return float(np.average((a - mu) ** 2, weights=ww))

        v0 = wvar(v[t == 0], w[t == 0])
        out[c] = wvar(v[t == 1], w[t == 1]) / v0 if v0 > 0 else np.nan
    return pd.Series(out)


def effective_sample_size(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / np.sum(w**2))


def balance_table(x: pd.DataFrame, t: np.ndarray, e: np.ndarray, pairs: dict[str, np.ndarray]) -> pd.DataFrame:
    w_ipw = ipw_weights(t, e, "ate")
    w_m = matched_frequency_weights(t, pairs)
    tab = pd.DataFrame(
        {
            "smd_raw": smd(x, t),
            "smd_ipw": smd(x, t, w_ipw),
            "smd_matched": smd(x, t, w_m),
            "var_ratio_raw": variance_ratio(x, t),
            "var_ratio_ipw": variance_ratio(x, t, w_ipw),
            "var_ratio_matched": variance_ratio(x, t, w_m),
        }
    )
    tab.index.name = "covariate"
    return tab


def run_propensity(df: pd.DataFrame, confounders: list[str], outcomes: list[str], n_boot: int, seed: int) -> dict[str, object]:
    """Point estimates, nonparametric individual-level bootstrap CIs (PS refit in every replicate) and diagnostics."""
    x = df[confounders]
    t = df["treated"].to_numpy()
    e = fit_propensity(x, t)
    pairs = match_pairs(t, e)
    est = {}
    for yname in outcomes:
        y = df[yname].to_numpy(dtype=float)
        est[yname] = {
            "ipw_ate": ipw_estimate(y, t, e, "ate"),
            "ipw_att": ipw_estimate(y, t, e, "att"),
            "match_ate": matching_estimate(y, t, pairs, "ate"),
            "match_att": matching_estimate(y, t, pairs, "att"),
            "ipw_mean_y0": ipw_mean_y0(y, t, e),
        }
    rng = np.random.default_rng(seed)
    boots: dict[str, dict[str, list[float]]] = {y: {k: [] for k in est[y]} for y in outcomes}
    n = len(df)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        xb, tb = x.iloc[ix], t[ix]
        eb = fit_propensity(xb, tb)
        pb = match_pairs(tb, eb)
        for yname in outcomes:
            yb = df[yname].to_numpy(dtype=float)[ix]
            boots[yname]["ipw_ate"].append(ipw_estimate(yb, tb, eb, "ate"))
            boots[yname]["ipw_att"].append(ipw_estimate(yb, tb, eb, "att"))
            boots[yname]["match_ate"].append(matching_estimate(yb, tb, pb, "ate"))
            boots[yname]["match_att"].append(matching_estimate(yb, tb, pb, "att"))
            boots[yname]["ipw_mean_y0"].append(ipw_mean_y0(yb, tb, eb))
    results: dict[str, dict[str, dict[str, float]]] = {}
    for yname in outcomes:
        results[yname] = {}
        for k, point in est[yname].items():
            b = np.asarray(boots[yname][k])
            se = float(b.std(ddof=1)) if len(b) > 1 else float("nan")
            results[yname][k] = {
                "estimate": point,
                "se": se,
                "ci_low": float(point - 1.96 * se),
                "ci_high": float(point + 1.96 * se),
            }
    bal = balance_table(x, t, e, pairs)
    w_ipw = ipw_weights(t, e, "ate")
    diag = {
        "ps_min": float(e.min()),
        "ps_max": float(e.max()),
        "share_ps_outside_0.05_0.95": float(np.mean((e < 0.05) | (e > 0.95))),
        "max_abs_smd_raw": float(bal["smd_raw"].abs().max()),
        "max_abs_smd_ipw": float(bal["smd_ipw"].abs().max()),
        "max_abs_smd_matched": float(bal["smd_matched"].abs().max()),
        "ess_treated_ipw": effective_sample_size(w_ipw[t == 1]),
        "ess_control_ipw": effective_sample_size(w_ipw[t == 0]),
        "n_treated": int(t.sum()),
        "n_control": int((1 - t).sum()),
        "share_matched_within_caliper": float(pairs["within"].mean()),
        "caliper_logit_ps": float(pairs["caliper"]),
        "n_boot": n_boot,
    }
    return {"estimates": results, "balance": bal, "diagnostics": diag, "propensity": e}
