"""Survival analysis: Kaplan-Meier, Cox PH (+ Schoenfeld diagnostics, stratification), RSF, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from .data import CAUSES, CONTRACTS, FEATURES, design_matrix

CONTRACT_DUMMIES = ["contract_one_year", "contract_two_year"]
STRAT_FEATURES = [f for f in FEATURES if f not in CONTRACT_DUMMIES]


def survival_frame(panel: pd.DataFrame, calendar_cutoff: int | None = None) -> pd.DataFrame:
    """Model frame with duration/event; campaign-treated customers are censored at the campaign.

    Treatment was randomised, so censoring treated customers at the campaign start is independent
    censoring and keeps the survival curves describing the *untreated* churn process.
    ``calendar_cutoff`` additionally censors everyone at that calendar month (information that
    would have been available at decision time).
    """
    dur = panel["duration"].to_numpy().astype(float)
    ev = panel["event"].to_numpy().astype(int)
    tac = panel["tenure_at_campaign"].to_numpy()
    treated = panel["treatment"].to_numpy() == 1
    cut = treated & (dur > tac)
    dur = np.where(cut, tac, dur)
    ev = np.where(cut, 0, ev)
    if calendar_cutoff is not None:
        max_t = calendar_cutoff - panel["start_month"].to_numpy()
        over = dur > max_t
        dur = np.where(over, max_t, dur)
        ev = np.where(over, 0, ev)
    out = design_matrix(panel)
    out["contract"] = panel["contract"].to_numpy()
    out["duration"] = dur
    out["event"] = ev
    out["cause"] = np.where(ev == 1, panel["cause"].to_numpy(), "censored")
    out = out[out["duration"] > 0]
    return out


def kaplan_meier_by(sf: pd.DataFrame, column: str, labels: dict | None = None) -> tuple[dict, float]:
    """KM curves per level of ``column`` plus the multivariate log-rank p-value."""
    curves = {}
    for level, g in sf.groupby(column):
        km = KaplanMeierFitter().fit(g["duration"], g["event"])
        name = labels.get(level, str(level)) if labels else str(level)
        curves[name] = km
    lr = multivariate_logrank_test(sf["duration"], sf[column], sf["event"])
    return curves, float(lr.p_value)


def km_table(curves: dict, times: list[int]) -> pd.DataFrame:
    rows = []
    for name, km in curves.items():
        s = km.survival_function_at_times(times).to_numpy()
        rows.append(
            {"segment": name, **{f"S({t})": round(float(v), 4) for t, v in zip(times, s)}, "median": km.median_survival_time_}
        )
    return pd.DataFrame(rows)


def fit_cox(sf: pd.DataFrame, stratified: bool, penalizer: float = 0.0) -> CoxPHFitter:
    cols = (STRAT_FEATURES + ["contract"]) if stratified else FEATURES
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(sf[cols + ["duration", "event"]], "duration", "event", strata=["contract"] if stratified else None)
    return cph


def ph_test(cph: CoxPHFitter, sf: pd.DataFrame, stratified: bool) -> pd.DataFrame:
    cols = (STRAT_FEATURES + ["contract"]) if stratified else FEATURES
    res = proportional_hazard_test(cph, sf[cols + ["duration", "event"]], time_transform="rank")
    out = res.summary[["test_statistic", "p"]].reset_index()
    out.columns = ["covariate", "test_statistic", "p_value"]
    return out


def scaled_schoenfeld(cph: CoxPHFitter, sf: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    r = cph.compute_residuals(sf[FEATURES + ["duration", "event"]], "scaled_schoenfeld")
    t = sf.loc[r.index, "duration"].to_numpy()
    out = r[covariates].copy()
    out["duration"] = t
    return out


def cox_survival_matrix(cph: CoxPHFitter, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """S(t | x) for each row of X at ``times`` -> array (n, len(times))."""
    sf = cph.predict_survival_function(X, times=times)
    return sf[X.index].to_numpy().T  # stratified predictions come back grouped by stratum


def fit_rsf(sf: pd.DataFrame, n_trees: int, min_leaf: int, seed: int) -> RandomSurvivalForest:
    y = Surv.from_arrays(sf["event"].astype(bool), sf["duration"])
    rsf = RandomSurvivalForest(
        n_estimators=n_trees, min_samples_leaf=min_leaf, max_features="sqrt", n_jobs=-1, random_state=seed, low_memory=False
    )
    rsf.fit(sf[FEATURES].to_numpy(), y)
    return rsf


def rsf_survival_matrix(rsf: RandomSurvivalForest, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    fns = rsf.predict_survival_function(X[FEATURES].to_numpy(), return_array=False)
    return np.vstack([f(times) for f in fns])


def evaluate(train: pd.DataFrame, test: pd.DataFrame, surv: np.ndarray, times: np.ndarray, c_time: int) -> dict:
    """Harrell C, Uno C, time-dependent AUC and integrated Brier score from S(t|x) at ``times``."""
    # Truncate *test* follow-up just past the last evaluation time: everyone still at risk at month 36
    # is administratively censored, so the IPCW censoring survival is zero there.
    tau = float(times.max()) + 3

    def _y(d: pd.DataFrame):
        over = d["duration"].to_numpy() > tau
        return Surv.from_arrays(d["event"].astype(bool).to_numpy() & ~over, np.minimum(d["duration"].to_numpy(), tau))

    y_tr = Surv.from_arrays(train["event"].astype(bool).to_numpy(), train["duration"].to_numpy())
    y_te = _y(test)
    i_c = int(np.where(times == c_time)[0][0])
    risk_c = 1 - surv[:, i_c]
    harrell = concordance_index_censored(test["event"].astype(bool).to_numpy(), test["duration"].to_numpy(), risk_c)[0]
    uno = concordance_index_ipcw(y_tr, y_te, risk_c, tau=float(times.max()))[0]
    auc_t, mean_auc = cumulative_dynamic_auc(y_tr, y_te, 1 - surv, times)
    ibs = integrated_brier_score(y_tr, y_te, surv, times)
    return {
        "harrell_c": float(harrell),
        "uno_c": float(uno),
        "td_auc": {int(t): float(a) for t, a in zip(times, auc_t)},
        "td_auc_mean": float(mean_auc),
        "integrated_brier": float(ibs),
    }


def aalen_johansen(sf: pd.DataFrame, max_t: int) -> pd.DataFrame:
    """Cumulative incidence per cause (discrete-time Aalen-Johansen, exact for monthly data)."""
    t = sf["duration"].to_numpy().astype(int)
    cause = sf["cause"].to_numpy()
    s_prev = 1.0
    cif = {c: 0.0 for c in CAUSES}
    rows = []
    for m in range(1, max_t + 1):
        at_risk = (t >= m).sum()
        if at_risk == 0:
            break
        at_m = t == m
        d = {c: int((at_m & (cause == c)).sum()) for c in CAUSES}
        for c in CAUSES:
            cif[c] += s_prev * d[c] / at_risk
        s_prev *= 1 - sum(d.values()) / at_risk
        rows.append({"month": m, "at_risk": int(at_risk), "overall_survival": s_prev, **{f"cif_{c}": cif[c] for c in CAUSES}})
    return pd.DataFrame(rows)


CONTRACT_LABELS = dict(enumerate(CONTRACTS))
