"""Procedural re-implementation of the legacy SAS programs and Excel workbook.

This is the *reference* the modern pipeline is reconciled against. It deliberately
mimics how the legacy code works rather than how it should work:

* SAS DATA steps run row by row over sorted data with BY-group FIRST./LAST. flags
  and RETAINed variables, and SAS missing-value arithmetic (missing + x = missing,
  while PROC SUMMARY's SUM ignores missing).
* ``PROC SORT NODUPKEY`` keeps the first record per key in *physical file order*,
  so the result depends on how the extract happened to be ordered.
* Excel formulas: ``ROUND`` (half away from zero) applied cell by cell,
  ``VLOOKUP(..., TRUE)`` approximate match and ``IFERROR(..., 0)``.

Source files: ``legacy/sas/*.sas`` and ``legacy/excel/QRT_build.bas``. They are
committed as the documentation of record and are never executed here.
"""

from __future__ import annotations

from bisect import bisect_right

import pandas as pd

from .config import ELR_BP, IBNR_FROM_QUARTER, LOBS, QUARTERS
from .generator import Extracts

MISSING = None  # SAS "." for numerics


def sas_round_div(num: int, den: int) -> int:
    """SAS ROUND(num/den, 1) / Excel ROUND(num/den, 0): half away from zero (den > 0)."""
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def sas_add(a, b):
    """SAS ``a + b``: missing if either operand is missing."""
    return MISSING if a is MISSING or b is MISSING else a + b


def sas_sub(a, b):
    return MISSING if a is MISSING or b is MISSING else a - b


def excel_vlookup_approx(value: str, table: dict[str, int]) -> int:
    """Excel VLOOKUP(value, table, 2, TRUE) on a table sorted A-Z.

    Returns the row for the largest key <= value, so an unmapped code silently
    picks up its alphabetical predecessor's value instead of #N/A.
    """
    keys = sorted(table)
    i = bisect_right(keys, value)
    if i == 0:
        raise ValueError("#N/A")
    return table[keys[i - 1]]


# --------------------------------------------------------------------------- 01_policy_prep.sas
def step01_policy_prep(policy_versions: pd.DataFrame) -> dict[int, tuple[str, str, int]]:
    """PROC SORT NODUPKEY BY policy_id, then the LOB mapping DATA step."""
    rows = policy_versions.sort_values("load_seq").itertuples(index=False)  # physical file order
    # PROC SORT is stable: within a BY value the first record in file order survives.
    seen: dict[int, tuple[str, str, int]] = {}
    for r in rows:
        if r.policy_id in seen:
            continue  # NODUPKEY drops it
        # DATA step: IF lob_code_raw IN ('COMMERCIAL','HOME',...) THEN lob = lob_code_raw; ELSE lob = 'OTHER';
        lob = r.lob_code_raw if r.lob_code_raw in LOBS else "OTHER"  # case- and blank-sensitive
        seen[r.policy_id] = (lob, r.region, r.version_no)
    return seen


# --------------------------------------------------------------------------- 02_premium_earn.sas
def step02_premium_earn(premium_txns: pd.DataFrame, pol: dict[int, tuple[str, str, int]], close_q: int) -> pd.DataFrame:
    """MERGE txns with POL_DEDUP BY policy_id; DO q = 1 TO &close; earn pro rata; ROUND to the cent."""
    out = []
    quarters = QUARTERS[:close_q]
    txns = premium_txns.sort_values(["policy_id", "txn_id"])
    for t in txns.itertuples(index=False):
        if t.policy_id not in pol:  # IF a AND b
            continue
        lob, region, _ = pol[t.policy_id]
        term = t.txn_exp - t.txn_eff
        for q in quarters:
            days = max(0, min(t.txn_exp, q.end + 1) - max(t.txn_eff, q.start))
            written = t.premium_cents if q.start <= t.txn_eff <= q.end else 0
            if days == 0 and written == 0:
                continue
            earned = sas_round_div(t.premium_cents * days, term)  # earned = ROUND(prem*days/term, 1);
            out.append((t.txn_id, q.idx, t.policy_id, lob, region, written, earned * 10_000))
    return pd.DataFrame(out, columns=["txn_id", "quarter", "policy_id", "lob", "region", "written_cents", "earned_ucents"])


# --------------------------------------------------------------------------- 03_claims.sas
def step03_claims(claim_valuations: pd.DataFrame, pol: dict[int, tuple[str, str, int]]) -> pd.DataFrame:
    """BY claim_id valuation_q with RETAIN prev_paid prev_inc and SAS missing arithmetic."""
    out = []
    vals = claim_valuations.sort_values(["claim_id", "valuation_q"])
    prev_claim = None
    prev_paid = 0
    prev_inc = 0
    for v in vals.itertuples(index=False):
        if v.policy_id not in pol:
            continue
        first = v.claim_id != prev_claim  # FIRST.claim_id
        if first:
            prev_paid, prev_inc = 0, 0
        reserve = MISSING if pd.isna(v.case_reserve_cents) else int(v.case_reserve_cents)
        incurred_cum = sas_add(int(v.cum_paid_cents), reserve)  # incurred_cum = cum_paid + case_reserve;
        paid_q = int(v.cum_paid_cents) - prev_paid
        incurred_q = sas_sub(incurred_cum, prev_inc)
        lob, region, _ = pol[v.policy_id]
        out.append((v.claim_id, v.valuation_q, v.policy_id, lob, region, paid_q, reserve, incurred_cum, incurred_q))
        prev_paid, prev_inc = int(v.cum_paid_cents), incurred_cum  # RETAIN
        prev_claim = v.claim_id
    df = pd.DataFrame(
        out, columns=["claim_id", "quarter", "policy_id", "lob", "region", "paid_cents", "case_reserve_cents", "incurred_cum_cents", "incurred_cents"]
    )
    for c in ["case_reserve_cents", "incurred_cum_cents", "incurred_cents"]:
        df[c] = df[c].astype("Int64")
    return df


# --------------------------------------------------------------------------- 04_summarise.sas
def step04_summarise(prem: pd.DataFrame, clm: pd.DataFrame) -> list[dict]:
    """PROC SUMMARY NWAY by quarter lob region (SUM ignores missing), then MERGE the two summaries."""
    cells: dict[tuple, dict] = {}

    def cell(k):
        if k not in cells:
            cells[k] = {"written": 0, "earned_u": 0, "paid": 0, "reserve": 0, "incurred": 0, "pols": set()}
        return cells[k]

    for r in prem.itertuples(index=False):
        c = cell((r.quarter, r.lob, r.region))
        c["written"] += r.written_cents
        c["earned_u"] += r.earned_ucents
        if r.earned_ucents > 0:
            c["pols"].add(r.policy_id)  # PROC SORT NODUPKEY BY quarter lob region policy_id
    for r in clm.itertuples(index=False):
        c = cell((r.quarter, r.lob, r.region))
        c["paid"] += r.paid_cents
        if not pd.isna(r.case_reserve_cents):
            c["reserve"] += int(r.case_reserve_cents)
        if not pd.isna(r.incurred_cents):
            c["incurred"] += int(r.incurred_cents)
    return [{"key": k, **v} for k, v in sorted(cells.items())]


# --------------------------------------------------------------------------- QRT_build.bas (Excel)
def step05_excel(summary: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summary:
        q, lob, region = s["key"]
        earned = sas_round_div(s["earned_u"], 10_000)
        incurred = s["incurred"]
        # =IFERROR(ROUND(incurred/earned, 4), 0)
        lr_bp = sas_round_div(incurred * 10_000, earned) if earned > 0 else 0
        ibnr = None
        if q >= IBNR_FROM_QUARTER:
            # =MAX(0, ROUND(earned * VLOOKUP(lob, ELR, 2, TRUE), 0) - incurred)
            elr = excel_vlookup_approx(lob, ELR_BP)
            ibnr = max(0, sas_round_div(earned * elr, 10_000) - incurred)
        rows.append(
            {
                "quarter": QUARTERS[q - 1].label,
                "lob": lob,
                "region": region,
                "written_premium_cents": s["written"],
                "earned_premium_cents": earned,
                "claims_paid_cents": s["paid"],
                "case_reserve_cents": s["reserve"],
                "incurred_cents": incurred,
                "loss_ratio_bp": lr_bp,
                "ibnr_cents": ibnr,
                "policies_in_force": len(s["pols"]),
            }
        )
    df = pd.DataFrame(rows)
    df["loss_ratio_bp"] = df["loss_ratio_bp"].astype("Int64")
    df["ibnr_cents"] = df["ibnr_cents"].astype("Int64")
    return df


def run_legacy(ex: Extracts, close_q: int) -> dict[str, pd.DataFrame]:
    """Run the full legacy chain as at the close of quarter ``close_q``."""
    pol = step01_policy_prep(ex.policy_versions)
    prem = step02_premium_earn(ex.premium_txns, pol, close_q)
    clm = step03_claims(ex.claim_valuations, pol)
    report = step05_excel(step04_summarise(prem, clm))
    return {"premium_detail": prem, "claims_detail": clm, "report": report}
