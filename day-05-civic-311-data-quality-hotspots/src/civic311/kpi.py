"""Data-quality and operational KPIs over time, with robust anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ANOMALY_MIN_FAILS,
    ANOMALY_MIN_LIFT,
    ANOMALY_WINDOW,
    ANOMALY_Z,
    EXTRACT_TS,
    SLO_PASS_TARGET,
    VOLUME_Z,
    WINDOW_START,
)
from .contract import RULES


def daily_quality(df: pd.DataFrame, rejects: pd.DataFrame) -> pd.DataFrame:
    """One row per created date: rows, rows passing the contract, pass rate and each rule's failure rate.

    Rows whose created timestamp is missing, unparseable or outside the window cannot be dated; they are
    counted in the `undated` summary instead (see `undated_summary`).
    """
    dated = df.created_ts.between(pd.Timestamp(WINDOW_START), pd.Timestamp(EXTRACT_TS), inclusive="left")
    day = df.loc[dated, ["source_line"]].assign(date=df.loc[dated, "created_ts"].dt.normalize())
    bad_lines = set(rejects.source_line)
    day["failed"] = day.source_line.isin(bad_lines)
    out = day.groupby("date").agg(rows=("source_line", "size"), failed=("failed", "sum"))
    out["passed"] = out.rows - out.failed
    out["pass_rate"] = out.passed / out.rows
    r = rejects[["source_line", "reason_code"]].merge(day[["source_line", "date"]], on="source_line")
    wide = r.groupby(["date", "reason_code"]).size().unstack(fill_value=0)
    for rule in RULES:
        out[f"rate_{rule.code}"] = wide.get(rule.code, pd.Series(0, index=wide.index)).reindex(out.index).fillna(0)
        out[f"rate_{rule.code}"] /= out.rows
    full = pd.date_range(WINDOW_START, periods=365, freq="D")
    return out.reindex(full).rename_axis("date")


def robust_rate_anomalies(
    rate: pd.Series,
    n: pd.Series,
    window: int = ANOMALY_WINDOW,
    z_thr: float = ANOMALY_Z,
    min_lift: float = ANOMALY_MIN_LIFT,
    min_fails: int = ANOMALY_MIN_FAILS,
) -> pd.DataFrame:
    """Flag days whose rate is far above a trailing robust baseline (and with at least `min_fails` failing rows).

    baseline = median of the previous `window` days. scale = max(1.4826 * MAD, binomial sd), where the binomial sd
    uses the larger of the baseline and the pooled trailing rate (failures / rows over the window), so rare rules
    whose daily median is 0 are not flagged for a single failure on a low-volume day.
    """
    prev = rate.shift(1).rolling(window, min_periods=window // 2)
    med = prev.median()
    mad = (rate.shift(1) - med).abs().rolling(window, min_periods=window // 2).median()
    fails = (rate * n).shift(1).rolling(window, min_periods=window // 2).sum()
    rows = n.shift(1).rolling(window, min_periods=window // 2).sum()
    p_ref = np.maximum(med, (fails + 0.5) / rows).clip(1e-6, 1 - 1e-6)
    binom = np.sqrt(p_ref * (1 - p_ref) / n.clip(lower=1))
    scale = np.maximum(1.4826 * mad, binom)
    z = (rate - med) / scale
    flag = (z >= z_thr) & ((rate - med) >= min_lift) & (rate * n >= min_fails)
    return pd.DataFrame({"rate": rate, "baseline": med, "z": z, "flag": flag.fillna(False)})


def quality_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    """Run the detector on the overall failure rate and on every rule's daily rate; return flagged days."""
    series = {"ALL_RULES (1 - pass rate)": 1 - daily.pass_rate}
    series |= {r.code: daily[f"rate_{r.code}"] for r in RULES}
    rows = []
    for name, s in series.items():
        a = robust_rate_anomalies(s, daily.rows)
        for d, row in a[a.flag].iterrows():
            rows.append({"date": d, "kpi": name, "rate": row.rate, "baseline": row.baseline, "z": row.z})
    return pd.DataFrame(rows, columns=["date", "kpi", "rate", "baseline", "z"])


def score_incidents(anoms: pd.DataFrame, incidents: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Day-level evaluation of the quality anomaly detector against the injected incidents."""
    inc_days = {}
    for inc in incidents:
        for d in pd.date_range(inc["start"], inc["end"], freq="D"):
            inc_days[d] = inc
    rule_flags = anoms[anoms.kpi != "ALL_RULES (1 - pass rate)"]
    per = []
    for inc in incidents:
        days = pd.date_range(inc["start"], inc["end"], freq="D")
        hit = rule_flags[(rule_flags.kpi == _rule_for(inc["defect"])) & rule_flags.date.isin(days)]
        per.append(
            {
                "incident": inc["id"], "start": inc["start"], "end": inc["end"], "what": inc["what"],
                "rule": _rule_for(inc["defect"]), "days": len(days), "days_flagged": hit.date.nunique(),
                "max_z": round(float(hit.z.max()), 1) if len(hit) else None,
            }
        )  # fmt: skip
    flagged_days = set(rule_flags.date)
    false_days = sorted(d for d in flagged_days if d not in inc_days)
    summary = {
        "incident_days": len(inc_days),
        "incident_days_flagged": int(sum(r["days_flagged"] for r in per)),
        "incidents_detected": int(sum(r["days_flagged"] > 0 for r in per)),
        "incidents": len(incidents),
        "false_alarm_days": len(false_days),
        "false_alarm_dates": [str(d.date()) for d in false_days],
    }
    return pd.DataFrame(per), summary


def _rule_for(defect: str) -> str:
    return {"COORDS_MISSING": "COORDS_MISSING", "CLOSED_BEFORE_CREATED": "CLOSED_BEFORE_CREATED",
            "WARD_INVALID": "WARD_INVALID"}[defect]  # fmt: skip


def volume_anomalies(df: pd.DataFrame, by_category: bool = True) -> pd.DataFrame:
    """Operational KPI: daily request volume vs the median of the same weekday over the previous 4 weeks.

    z = log(count / baseline) / scale, scale = max(1.4826 * MAD of the log ratios over the year, 1/sqrt(baseline)).
    """
    dated = df.created_ts.between(pd.Timestamp(WINDOW_START), pd.Timestamp(EXTRACT_TS), inclusive="left")
    d = df.loc[dated].assign(date=df.loc[dated, "created_ts"].dt.normalize())
    groups: dict[str, pd.Series] = {"ALL": d.groupby("date").size()}
    if by_category:
        for c, g in d.dropna(subset=["category"]).groupby("category"):
            groups[c] = g.groupby("date").size()
    full = pd.date_range(WINDOW_START, periods=365, freq="D")
    rows = []
    for name, s in groups.items():
        s = s.reindex(full, fill_value=0).astype(float)
        lags = pd.concat([s.shift(7 * k) for k in range(1, 5)], axis=1)
        base = lags.median(axis=1).where(lags.notna().sum(axis=1) >= 3)
        lr = np.log((s + 1) / (base + 1))
        mad = (lr - lr.median()).abs().median()
        scale = np.maximum(1.4826 * mad, 1 / np.sqrt(base.clip(lower=1)))
        z = lr / scale
        for day in z.index[(z >= VOLUME_Z).fillna(False)]:
            rows.append({"date": day, "series": name, "count": int(s[day]), "baseline": float(base[day]),
                         "ratio": float(s[day] / max(base[day], 1)), "z": float(z[day])})  # fmt: skip
    return pd.DataFrame(rows, columns=["date", "series", "count", "baseline", "ratio", "z"])


def slo_burndown(daily: pd.DataFrame, target: float = SLO_PASS_TARGET) -> pd.DataFrame:
    """Per-quarter error budget: (1 - target) x rows in the quarter; track the share of budget left each day."""
    d = daily.dropna(subset=["rows"]).copy()
    d["quarter"] = d.index.to_period("Q").astype(str)
    out = []
    for q, g in d.groupby("quarter"):
        budget = (1 - target) * g.rows.sum()
        spent = g.failed.cumsum()
        elapsed = np.arange(1, len(g) + 1) / len(g)
        out.append(
            pd.DataFrame(
                {
                    "date": g.index, "quarter": q, "budget_rows": budget, "failed_cum": spent.to_numpy(),
                    "budget_left": 1 - spent.to_numpy() / budget, "ideal_left": 1 - elapsed,
                }
            )
        )  # fmt: skip
    return pd.concat(out, ignore_index=True)
