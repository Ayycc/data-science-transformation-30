"""Optional replication on the PUBLIC LaLonde (1986) / Dehejia-Wahba NSW data.

Benchmark: the experimental difference in means (NSW treated vs NSW randomised controls).
Observational design: the same NSW treated against the PSID comparison group (``MatchIt::lalonde``),
which is where naive and regression estimates famously go wrong.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATA_RAW, DATA_SAMPLE, LALONDE_URLS
from .propensity import fit_propensity, ipw_estimate, smd

COVS = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "u74", "u75"]


def _download(url: str, timeout: float = 20.0) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
        return pd.read_csv(io.BytesIO(r.read()))


def load_lalonde(prefer: str = "auto") -> tuple[dict[str, pd.DataFrame] | None, str]:
    """Return ({'nsw_experimental', 'nsw_psid'}, path) where path is download | cache | sample | unavailable."""
    names = list(LALONDE_URLS)
    if prefer in ("auto", "download"):
        try:
            frames = {k: _download(LALONDE_URLS[k]) for k in names}
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            for k, f in frames.items():
                f.to_csv(DATA_RAW / f"{k}.csv", index=False)
            return frames, "download"
        except Exception:  # noqa: BLE001 - any network failure means fall back
            if prefer == "download":
                return None, "unavailable"
    for label, folder in (("cache", DATA_RAW), ("sample", DATA_SAMPLE)):
        paths = [Path(folder) / f"{k}.csv" for k in names]
        if all(p.exists() for p in paths):
            return {k: pd.read_csv(p) for k, p in zip(names, paths, strict=True)}, label
    return None, "unavailable"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "race" in d.columns:
        d["black"] = (d["race"] == "black").astype(int)
        d["hispan"] = (d["race"] == "hispan").astype(int)
    if "hisp" in d.columns and "hispan" not in d.columns:
        d["hispan"] = d["hisp"]
    if "marr" in d.columns and "married" not in d.columns:
        d["married"] = d["marr"]
    d["u74"] = (d["re74"] == 0).astype(int)
    d["u75"] = (d["re75"] == 0).astype(int)
    return d


def dr_att_crossfit(x: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> tuple[float, float]:
    """Cross-fitted doubly robust ATT (Chernozhukov et al. 2018 score) with random-forest nuisances."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import StratifiedKFold

    e = np.zeros(len(t))
    m0 = np.zeros(len(t))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, t):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=seed).fit(x[tr], t[tr])
        e[te] = clf.predict_proba(x[te])[:, 1]
        c = tr[t[tr] == 0]
        reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=seed).fit(x[c], y[c])
        m0[te] = reg.predict(x[te])
    e = np.clip(e, 0.01, 0.99)
    p = t.mean()
    score = (t - (1 - t) * e / (1 - e)) * (y - m0) / p
    att = float(score.mean())
    psi = score - t * att / p
    return att, float(psi.std(ddof=1) / np.sqrt(len(t)))


def run_lalonde(frames: dict[str, pd.DataFrame], seed: int, n_boot: int = 200) -> dict[str, object]:
    exp = _prep(frames["nsw_experimental"])
    if "data_id" in exp.columns:
        exp = exp[exp["data_id"].astype(str).str.contains("Dehejia")]
    t_e = exp["treat"].to_numpy()
    y_e = exp["re78"].to_numpy(dtype=float)
    a, b = y_e[t_e == 1], y_e[t_e == 0]
    bench = float(a.mean() - b.mean())
    se_b = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))

    obs = _prep(frames["nsw_psid"])
    t = obs["treat"].to_numpy()
    y = obs["re78"].to_numpy(dtype=float)
    x = obs[COVS].astype(float)
    naive = float(y[t == 1].mean() - y[t == 0].mean())

    import statsmodels.api as sm

    ols = sm.OLS(y, sm.add_constant(pd.concat([pd.Series(t, name="treat", index=x.index), x], axis=1))).fit(cov_type="HC1")
    ols_b, ols_se = float(ols.params["treat"]), float(ols.bse["treat"])

    e = fit_propensity(x, t)
    ipw_att = ipw_estimate(y, t, e, "att")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(t), len(t))
        eb = fit_propensity(x.iloc[ix], t[ix])
        boots.append(ipw_estimate(y[ix], t[ix], eb, "att"))
    ipw_se = float(np.std(boots, ddof=1))

    dr_att, dr_se = dr_att_crossfit(x.to_numpy(), t, y, seed)

    def row(est: float, se: float | None, lo_: float | None = None, hi_: float | None = None) -> dict[str, float]:
        if lo_ is None and se is not None:
            lo_, hi_ = est - 1.96 * se, est + 1.96 * se
        r = {"estimate": est, "ci_low": lo_, "ci_high": hi_, "bias_vs_benchmark": est - bench}
        r["benchmark_in_ci"] = bool(lo_ is not None and lo_ <= bench <= hi_)
        return r

    return {
        "n_experimental": int(len(exp)),
        "n_observational": int(len(obs)),
        "n_treated": int(t.sum()),
        "n_psid_controls": int((1 - t).sum()),
        "experimental_benchmark": {"estimate": bench, "ci_low": bench - 1.96 * se_b, "ci_high": bench + 1.96 * se_b},
        "naive_psid": row(naive, None, None, None),
        "ols_adjusted_psid": row(ols_b, ols_se),
        "ipw_att_psid": row(ipw_att, ipw_se),
        "dr_att_crossfit_psid": row(dr_att, dr_se),
        "max_abs_smd_raw": float(smd(x, t).abs().max()),
        "share_psid_ps_below_0.05": float(np.mean(e[t == 0] < 0.05)),
        "welch_p_benchmark": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
    }
