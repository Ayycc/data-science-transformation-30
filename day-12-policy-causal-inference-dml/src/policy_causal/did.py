"""Staggered difference-in-differences: Callaway & Sant'Anna (2021) group-time ATTs, event study,
pre-trend test, and the two-way fixed-effects (TWFE) comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _wide(panel: pd.DataFrame, outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = panel.pivot(index="county_id", columns="quarter", values=outcome).sort_index()
    g = panel.groupby("county_id")["first_treated"].first().reindex(w.index).to_numpy()
    return w.to_numpy(), g, w.columns.to_numpy()


def _att_gt_matrix(y: np.ndarray, g: np.ndarray, quarters: np.ndarray, cohorts: list[int], control: str) -> dict[tuple[int, int], float]:
    """ATT(g,t) with a universal base period g-1 (long differences), unconditional parallel trends."""
    out: dict[tuple[int, int], float] = {}
    for coh in cohorts:
        b = coh - 1
        if b < quarters.min():
            continue
        treat = g == coh
        if treat.sum() == 0:
            continue
        for t in quarters:
            if t == b:
                continue
            if control == "never":
                ctrl = ~np.isfinite(g)
            else:  # not-yet-treated at max(t, b)
                ctrl = (g > max(t, b)) & (g != coh)
            if ctrl.sum() == 0:
                continue
            dy = y[:, t] - y[:, b]
            out[(coh, int(t))] = float(dy[treat].mean() - dy[ctrl].mean())
    return out


def _aggregate(att: dict[tuple[int, int], float], sizes: dict[int, int]) -> dict[str, object]:
    post = [(k, v) for k, v in att.items() if k[1] >= k[0]]
    wsum = sum(sizes[k[0]] for k, _ in post)
    overall = sum(sizes[k[0]] * v for k, v in post) / wsum if wsum else float("nan")
    ev: dict[int, list[tuple[float, int]]] = {}
    for (coh, t), v in att.items():
        ev.setdefault(t - coh, []).append((v, sizes[coh]))
    event = {e: sum(v * n for v, n in lst) / sum(n for _, n in lst) for e, lst in ev.items()}
    by_cohort = {}
    for coh in sizes:
        vals = [v for (c, t), v in att.items() if c == coh and t >= c]
        if vals:
            by_cohort[coh] = float(np.mean(vals))
    return {"overall": float(overall), "event": dict(sorted(event.items())), "cohort": by_cohort}


def callaway_santanna(
    panel: pd.DataFrame, outcome: str = "employment_rate", control: str = "never", n_boot: int = 499, seed: int = 0
) -> dict[str, object]:
    y, g, quarters = _wide(panel, outcome)
    cohorts = sorted(int(c) for c in np.unique(g[np.isfinite(g)]))
    sizes = {c: int((g == c).sum()) for c in cohorts}
    att = _att_gt_matrix(y, g, quarters, cohorts, control)
    agg = _aggregate(att, sizes)

    rng = np.random.default_rng(seed)
    n = len(g)
    boot_overall, boot_event = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        yb, gb = y[ix], g[ix]
        sb = {c: int((gb == c).sum()) for c in cohorts}
        if min(sb.values()) == 0 or (control == "never" and np.isfinite(gb).all()):
            continue
        ab = _aggregate(_att_gt_matrix(yb, gb, quarters, cohorts, control), sb)
        boot_overall.append(ab["overall"])
        boot_event.append(ab["event"])
    se_overall = float(np.std(boot_overall, ddof=1))
    events = sorted(agg["event"])
    bmat = np.array([[b.get(e, np.nan) for e in events] for b in boot_event])
    se_event = np.nanstd(bmat, axis=0, ddof=1)
    ev_df = pd.DataFrame({"event_time": events, "att": [agg["event"][e] for e in events], "se": se_event})
    ev_df["ci_low"] = ev_df["att"] - 1.96 * ev_df["se"]
    ev_df["ci_high"] = ev_df["att"] + 1.96 * ev_df["se"]

    # joint pre-trend Wald test on all pre-period event-time coefficients (e <= -2)
    pre_idx = [i for i, e in enumerate(events) if e <= -2]
    theta = ev_df["att"].to_numpy()[pre_idx]
    bpre = bmat[:, pre_idx]
    bpre = bpre[~np.isnan(bpre).any(axis=1)]
    cov = np.cov(bpre, rowvar=False)
    wald = float(theta @ np.linalg.pinv(cov) @ theta)
    pval = float(1 - stats.chi2.cdf(wald, df=len(pre_idx)))
    return {
        "control_group": control,
        "att_overall": agg["overall"],
        "att_overall_se": se_overall,
        "att_overall_ci": [agg["overall"] - 1.96 * se_overall, agg["overall"] + 1.96 * se_overall],
        "att_by_cohort": agg["cohort"],
        "event_study": ev_df,
        "att_gt": att,
        "pretrend_wald": wald,
        "pretrend_df": len(pre_idx),
        "pretrend_p": pval,
        "pretrend_max_abs": float(np.max(np.abs(theta))) if len(theta) else float("nan"),
        "n_boot_effective": len(boot_overall),
    }


def twfe(panel: pd.DataFrame, outcome: str = "employment_rate") -> dict[str, float]:
    """Static TWFE regression with county and quarter fixed effects, county-clustered SEs (linearmodels)."""
    from linearmodels.panel import PanelOLS

    p = panel.set_index(["county_id", "quarter"])
    mod = PanelOLS(p[outcome], p[["treated_now"]], entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    b = float(res.params["treated_now"])
    se = float(res.std_errors["treated_now"])
    return {"estimate": b, "se": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se}


def twfe_event_study(panel: pd.DataFrame, outcome: str = "employment_rate", window: tuple[int, int] = (-6, 10)) -> pd.DataFrame:
    """Dynamic TWFE with binned endpoints and e=-1 omitted (for comparison with CS)."""
    from linearmodels.panel import PanelOLS

    p = panel.copy()
    e = (p["quarter"] - p["first_treated"]).where(np.isfinite(p["first_treated"]))
    lo, hi = window
    eb = e.clip(lo, hi)
    cols = []
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        name = f"e_{k}".replace("-", "m")
        p[name] = (eb == k).astype(float).fillna(0.0)
        cols.append((k, name))
    p = p.set_index(["county_id", "quarter"])
    res = PanelOLS(p[outcome], p[[c for _, c in cols]], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    return pd.DataFrame(
        {
            "event_time": [k for k, _ in cols],
            "coef": [float(res.params[c]) for _, c in cols],
            "se": [float(res.std_errors[c]) for _, c in cols],
        }
    )
