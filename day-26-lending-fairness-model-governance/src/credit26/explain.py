"""Adverse-action reason codes from SHAP, and counterfactual 'what would change the decision' explanations."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import shap

from .data import DatasetSpec
from .models import TrainedGBM


def shap_values(m: TrainedGBM, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP in log-odds(bad) units. Rows sum (plus the base value) to the model's raw score."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex = shap.TreeExplainer(m.model)
        sv = ex.shap_values(m.X(df))
    if isinstance(sv, list):  # older SHAP returned [class0, class1]
        sv = sv[1]
    base = ex.expected_value
    base = float(np.ravel(base)[-1])
    return np.asarray(sv), base


def reason_codes(sv: np.ndarray, features: list[str], k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Rank features by their contribution towards 'bad' (largest positive SHAP first); return top-k names and values."""
    order = np.argsort(-sv, axis=1, kind="stable")[:, :k]
    names = np.array(features, dtype=object)[order]
    vals = np.take_along_axis(sv, order, axis=1)
    return names, vals


def code_table(spec: DatasetSpec, features: list[str]) -> dict[str, str]:
    return {f: f"R{i + 1:02d}" for i, f in enumerate(features)}


def reason_frame(df: pd.DataFrame, names: np.ndarray, vals: np.ndarray, spec: DatasetSpec, features: list[str]) -> pd.DataFrame:
    codes = code_table(spec, features)
    out = {"application_id": df["application_id"].to_numpy()}
    for j in range(names.shape[1]):
        out[f"reason_{j + 1}_code"] = [codes[n] for n in names[:, j]]
        out[f"reason_{j + 1}_shap"] = np.round(vals[:, j], 3)
    return pd.DataFrame(out)


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"


def _grid(x: np.ndarray, direction: int, bound: float, feature: str, K: int) -> np.ndarray:
    """(n, K) candidate values moving from x towards the bound (relative bounds for amounts/balances)."""
    frac = np.linspace(1.0 / K, 1.0, K)[None, :]
    x = x[:, None]
    if feature in ("loan_amount", "amount"):
        v = x * (1 - (1 - bound) * frac)
    elif feature == "savings_balance":
        v = x * (1 + (bound - 1) * frac)
    elif direction < 0:
        v = x - np.clip(x - bound, 0, None) * frac
    else:
        v = x + np.clip(bound - x, 0, None) * frac
    if feature in ("num_inquiries_6m",):
        v = np.floor(v)
    return v


def counterfactuals(
    m: TrainedGBM,
    declined: pd.DataFrame,
    cutoff: float,
    spec: DatasetSpec,
    scale: dict[str, float],
    K: int = 12,
    max_features: int = 2,
    batch: int = 250,
) -> pd.DataFrame:
    """Lowest-cost change of at most ``max_features`` actionable features that turns a decline into an approval.

    Cost = sum over changed features of |change| / IQR(feature, training data). Protected attributes and non-actionable
    history (delinquencies, history length, income, employment, address, geography) are never changed. The search is an
    exhaustive grid (K steps per feature, all single features and pairs), so the reported change is the cheapest on the grid.
    """
    act = [f for f in spec.actionable if f in m.features]
    combos: list[tuple] = [((f, k),) for f in act for k in range(K)]
    if max_features >= 2:
        for f1, f2 in itertools.combinations(act, 2):
            combos += [((f1, k1), (f2, k2)) for k1 in range(K) for k2 in range(K)]
    C = len(combos)
    kidx = {f: np.full(C, -1) for f in act}  # per combo: grid step for feature f, or -1 if unchanged
    for c, combo in enumerate(combos):
        for f, k in combo:
            kidx[f][c] = k
    p_before = m.p_bad(declined)
    recs = []
    for s0 in range(0, len(declined), batch):
        part = declined.iloc[s0 : s0 + batch]
        n = len(part)
        base = part[m.features].reset_index(drop=True)
        big = base.loc[base.index.repeat(C)].reset_index(drop=True)
        rows = np.arange(n).repeat(C)
        ci = np.tile(np.arange(C), n)
        cost = np.zeros(n * C)
        for f in act:
            g = _grid(part[f].to_numpy(dtype=float), *spec.actionable[f], f, K)
            orig = big[f].to_numpy(dtype=float)
            kk = kidx[f][ci]
            newv = np.where(kk >= 0, g[rows, np.clip(kk, 0, None)], orig)
            cost += np.abs(newv - orig) / scale[f]
            big[f] = newv
        p = m.p_bad(big)
        cost_ok = np.where((p <= cutoff) & (cost > 0), cost, np.inf).reshape(n, C)
        best = cost_ok.argmin(axis=1)
        for i in range(n):
            r = {
                "application_id": int(part["application_id"].iloc[i]),
                "found": bool(np.isfinite(cost_ok[i, best[i]])),
                "p_bad_before": float(p_before[s0 + i]),
            }
            if r["found"]:
                j = i * C + best[i]
                changes = []
                for ff, _ in combos[best[i]]:
                    a, b = float(part[ff].iloc[i]), float(big[ff].iloc[j])
                    if a != b:
                        changes.append(f"{ff}: {_fmt(a)} -> {_fmt(b)}")
                r.update({"n_changed": len(changes), "changes": "; ".join(changes), "cost": float(cost[j]), "p_bad_after": float(p[j])})
            recs.append(r)
    return pd.DataFrame(recs)
