"""Legacy-vs-modern reconciliation: datacompy row-by-row comparison plus step-wise attribution.

Method
------
1. ``COMPAT`` (all ModernConfig switches off) must equal the legacy output exactly
   on every row and column of all three outputs. Any difference is unexplained.
2. The intentional differences D1..D6 are switched on one at a time
   (cumulatively). Every cell that changes at step *k* must satisfy the documented
   predicate for D*k* (which rows, which columns, what magnitude). A changed cell that
   falls outside its predicate is counted as *unexplained*.
3. ``TARGET`` (all switches on) is compared with legacy with datacompy, and every
   mismatching cell is traced to the step that introduced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import datacompy
import numpy as np
import pandas as pd

from .config import LOBS, QUARTERS
from .generator import Extracts
from .modern import CLAIM_KEYS, PREM_KEYS, REPORT_KEYS

KEYS = {"premium_detail": PREM_KEYS, "claims_detail": CLAIM_KEYS, "report": REPORT_KEYS}
ROW = "__row__"


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("Int64", "int64"):
            out[c] = out[c].astype("Int64")
        else:
            out[c] = out[c].astype(object)
    return out


def diff_cells(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """All differing cells between two outputs; missing rows are reported with column ``__row__``."""
    a, b = _norm(a), _norm(b)
    m = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    parts = []
    only = m[m["_merge"] != "both"]
    if len(only):
        parts.append(pd.DataFrame({**{k: only[k].values for k in keys}, "column": ROW, "a": only["_merge"].astype(str).values, "b": None}))
    both = m[m["_merge"] == "both"]
    for c in [c for c in a.columns if c not in keys]:
        x, y = both[f"{c}_a"], both[f"{c}_b"]
        xn, yn = x.isna().to_numpy(), y.isna().to_numpy()
        xv = np.where(xn, "<NA>", x.astype(object).to_numpy())
        yv = np.where(yn, "<NA>", y.astype(object).to_numpy())
        neq = (xn != yn) | (~xn & ~yn & (xv != yv))
        if neq.any():
            sub = both[neq]
            parts.append(
                pd.DataFrame(
                    {**{k: sub[k].values for k in keys}, "column": c, "a": sub[f"{c}_a"].astype(object).values, "b": sub[f"{c}_b"].astype(object).values}
                )
            )
    if not parts:
        return pd.DataFrame(columns=[*keys, "column", "a", "b"])
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- affected sets from source data
def affected_sets(ex: Extracts, close_q: int) -> dict[str, set]:
    pv = ex.policy_versions
    code = pv["lob_code_raw"].astype(str)
    fixable = (~code.isin(LOBS)) & code.str.strip().str.upper().isin(LOBS)
    d4 = set(pv.loc[fixable, "policy_id"].tolist())

    first = pv.sort_values("load_seq").drop_duplicates("policy_id")[["policy_id", "version_no"]].rename(columns={"version_no": "legacy_v"})
    d2 = set()
    for q in QUARTERS[:close_q]:
        asof = pv[pv["version_eff"] <= q.end].groupby("policy_id")["version_no"].max().rename("asof_v").reset_index()
        j = asof.merge(first, on="policy_id")
        d2 |= {(p, q.idx) for p in j.loc[j["asof_v"] != j["legacy_v"], "policy_id"].tolist()}

    cv = ex.claim_valuations
    null_q = cv.loc[cv["case_reserve_cents"].isna(), ["claim_id", "valuation_q"]]
    d3 = {(c, q) for c, q in null_q.itertuples(index=False)} | {(c, q + 1) for c, q in null_q.itertuples(index=False)}
    return {"D4": d4, "D2": d2, "D3": d3}


def _cells_of(details: list[pd.DataFrame], mask_fn) -> set:
    out = set()
    for d in details:
        sub = d[mask_fn(d)]
        labels = sub["quarter"].map(lambda i: QUARTERS[int(i) - 1].label)
        out |= set(zip(labels, sub["lob"], sub["region"]))
    return out


@dataclass
class StepResult:
    step: str
    changed_cells: dict[str, int]
    changed_rows: dict[str, int]
    unexplained: dict[str, int]
    examples: list[dict] = field(default_factory=list)


def check_step(step: str, prev: dict[str, pd.DataFrame], cur: dict[str, pd.DataFrame], aff: dict[str, set]) -> StepResult:
    """Validate that every cell changed by switching on difference ``step`` matches its documented predicate."""
    diffs = {t: diff_cells(prev[t], cur[t], KEYS[t]) for t in KEYS}
    bad = {}
    examples = []
    for t, d in diffs.items():
        if d.empty:
            bad[t] = 0
            continue
        ok = np.zeros(len(d), dtype=bool)
        col = d["column"]
        if t == "premium_detail":
            pol = prev[t].set_index(["txn_id", "quarter"])["policy_id"]
            pids = [pol.get((a, b)) for a, b in zip(d["txn_id"], d["quarter"])]
            if step == "D4":
                ok = np.array([p in aff["D4"] for p in pids]) & col.eq("lob").to_numpy()
            elif step == "D2":
                ok = np.array([(p, int(q)) in aff["D2"] for p, q in zip(pids, d["quarter"])]) & col.isin(["lob", "region"]).to_numpy()
            elif step == "D1":
                is_earned = col.eq("earned_ucents").to_numpy()
                delta = np.array([abs(int(a) - int(b)) if e else np.inf for a, b, e in zip(d["a"], d["b"], is_earned)])
                ok = is_earned & (delta <= 5_000)
        elif t == "claims_detail":
            if step == "D4":
                pol = prev[t].set_index(["claim_id", "quarter"])["policy_id"]
                ok = np.array([pol.get((a, b)) in aff["D4"] for a, b in zip(d["claim_id"], d["quarter"])]) & col.eq("lob").to_numpy()
            elif step == "D2":
                pol = prev[t].set_index(["claim_id", "quarter"])["policy_id"]
                ok = np.array([(pol.get((a, b)), int(b)) in aff["D2"] for a, b in zip(d["claim_id"], d["quarter"])]) & col.isin(["lob", "region"]).to_numpy()
            elif step == "D3":
                ok = np.array([(int(a), int(b)) in aff["D3"] for a, b in zip(d["claim_id"], d["quarter"])])
                ok &= col.isin(["case_reserve_cents", "incurred_cum_cents", "incurred_cents"]).to_numpy()
        else:  # report
            cell = list(zip(d["quarter"], d["lob"], d["region"]))
            if step in ("D4", "D2", "D3"):
                det = [prev["premium_detail"], cur["premium_detail"], prev["claims_detail"], cur["claims_detail"]]
                if step == "D4":
                    allowed = _cells_of(det, lambda x: x["policy_id"].isin(aff["D4"]))
                elif step == "D2":
                    allowed = _cells_of(
                        det, lambda x: pd.Series([(p, int(q)) in aff["D2"] for p, q in zip(x["policy_id"], x["quarter"])], index=x.index, dtype=bool)
                    )
                else:
                    det = [prev["claims_detail"], cur["claims_detail"]]
                    allowed = _cells_of(
                        det, lambda x: pd.Series([(c, int(q)) in aff["D3"] for c, q in zip(x["claim_id"], x["quarter"])], index=x.index, dtype=bool)
                    )
                    ok = np.array([c in allowed for c in cell]) & col.isin(["incurred_cents", "loss_ratio_bp", "ibnr_cents"]).to_numpy()
                if step != "D3":
                    ok = np.array([c in allowed for c in cell])
            elif step == "D1":
                n_rows = prev["premium_detail"].assign(ql=prev["premium_detail"]["quarter"].map(lambda i: QUARTERS[int(i) - 1].label))
                n_rows = n_rows.groupby(["ql", "lob", "region"]).size().to_dict()
                earned_changed = set(
                    zip(
                        d.loc[col == "earned_premium_cents", "quarter"],
                        d.loc[col == "earned_premium_cents", "lob"],
                        d.loc[col == "earned_premium_cents", "region"],
                    )
                )
                oks = []
                for c, cname, a, b in zip(cell, col, d["a"], d["b"]):
                    if cname == "earned_premium_cents":
                        oks.append(abs(int(a) - int(b)) <= 0.5 * n_rows.get(c, 0) + 1)
                    elif cname in ("loss_ratio_bp", "ibnr_cents"):
                        oks.append(c in earned_changed)
                    else:
                        oks.append(False)
                ok = np.array(oks, dtype=bool)
            elif step == "D5":
                ok = (d["lob"].eq("OTHER") & col.eq("ibnr_cents")).to_numpy()
            elif step == "D6":
                ok = (col.eq("loss_ratio_bp") & d["a"].eq(0) & d["b"].isna()).to_numpy()
        bad[t] = int((~ok).sum())
        if (~ok).any():
            examples.extend(d[~ok].head(3).assign(table=t).to_dict("records"))
    changed_rows = {t: int(d.drop_duplicates(KEYS[t]).shape[0]) for t, d in diffs.items()}
    return StepResult(step, {t: len(d) for t, d in diffs.items()}, changed_rows, bad, examples)


def datacompy_compare(legacy: pd.DataFrame, modern: pd.DataFrame, keys: list[str]) -> datacompy.PandasCompare:
    def prep(df):
        out = df.copy()
        for c in out.columns:
            if str(out[c].dtype) in ("Int64", "int64"):
                out[c] = out[c].astype("float64")
            else:
                out[c] = out[c].astype(str)
        return out

    return datacompy.PandasCompare(prep(legacy), prep(modern), join_columns=keys, abs_tol=0, rel_tol=0, df1_name="legacy", df2_name="modern")


def compare_summary(cmp: datacompy.PandasCompare) -> dict:
    mism = sum(int(s["unequal_cnt"]) for s in cmp.column_stats)
    return {
        "matches": bool(cmp.matches()),
        "rows_legacy": int(len(cmp.df1)),
        "rows_modern": int(len(cmp.df2)),
        "rows_only_legacy": int(len(cmp.df1_unq_rows)),
        "rows_only_modern": int(len(cmp.df2_unq_rows)),
        "rows_in_common": int(len(cmp.intersect_rows)),
        "rows_all_columns_equal": int(cmp.count_matching_rows()),
        "unequal_cells": mism,
    }
