"""Demeaned synthetic control (Abadie, Diamond & Hainmueller 2010; Ferman & Pinto 2021) for a single pilot county, with placebo-in-space inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def sc_weights(y_treated_pre: np.ndarray, y_donors_pre: np.ndarray) -> np.ndarray:
    """Non-negative donor weights summing to one that minimise pre-period squared error."""
    k = y_donors_pre.shape[1]

    def loss(w: np.ndarray) -> float:
        r = y_treated_pre - y_donors_pre @ w
        return float(r @ r)

    def grad(w: np.ndarray) -> np.ndarray:
        return -2 * y_donors_pre.T @ (y_treated_pre - y_donors_pre @ w)

    res = minimize(
        loss,
        np.full(k, 1.0 / k),
        jac=grad,
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def _fit_one(y: np.ndarray, unit: int, donors: np.ndarray, t0: int) -> dict[str, object]:
    w = sc_weights(y[unit, :t0], y[donors, :t0].T)
    synth = y[donors].T @ w
    gap = y[unit] - synth
    pre_rmspe = float(np.sqrt(np.mean(gap[:t0] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[t0:] ** 2)))
    return {"weights": w, "synthetic": synth, "gap": gap, "pre_rmspe": pre_rmspe, "post_rmspe": post_rmspe, "ratio": post_rmspe / pre_rmspe}


def synthetic_control(panel: pd.DataFrame, treated_unit: int, outcome: str = "employment_rate") -> dict[str, object]:
    w = panel.pivot(index="county_id", columns="quarter", values=outcome).sort_index()
    first = panel.groupby("county_id")["first_treated"].first().reindex(w.index)
    y = w.to_numpy()
    ids = w.index.to_numpy()
    unit = int(np.flatnonzero(ids == treated_unit)[0])
    t0 = int(first.loc[treated_unit])
    donors = np.flatnonzero(~np.isfinite(first.to_numpy()))
    # demeaned synthetic control (Ferman & Pinto 2021): match pre-period *dynamics*, not levels
    level = y[:, :t0].mean(axis=1, keepdims=True)
    y = y - level
    main = _fit_one(y, unit, donors, t0)
    # placebo in space: pretend each donor was treated at t0, using the other donors
    ratios = []
    for d in donors:
        others = donors[donors != d]
        ratios.append(_fit_one(y, int(d), others, t0)["ratio"])
    ratios = np.asarray(ratios)
    p_value = float((1 + np.sum(ratios >= main["ratio"])) / (1 + len(ratios)))
    truth = panel[panel["county_id"] == treated_unit].sort_values("quarter")["true_effect"].to_numpy()
    top = np.argsort(-main["weights"])[:5]
    return {
        "treated_unit": int(treated_unit),
        "adoption_quarter": t0,
        "n_donors": int(len(donors)),
        "pre_rmspe": main["pre_rmspe"],
        "post_rmspe": main["post_rmspe"],
        "avg_post_gap": float(main["gap"][t0:].mean()),
        "true_avg_post_effect": float(truth[t0:].mean()),
        "placebo_p_value": p_value,
        "rmspe_ratio": main["ratio"],
        "top_donor_weights": {int(ids[donors[i]]): float(main["weights"][i]) for i in top},
        "series": pd.DataFrame(
            {
                "quarter": w.columns.to_numpy(),
                "observed": y[unit] + level[unit, 0],
                "synthetic": main["synthetic"] + level[unit, 0],
                "gap": main["gap"],
                "true_effect": truth,
            }
        ),
    }
