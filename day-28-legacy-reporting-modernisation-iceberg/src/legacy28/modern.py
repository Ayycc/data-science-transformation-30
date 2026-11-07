"""Modern implementation: engine-portable Ibis expressions.

The same expression builders run on DuckDB and Polars. Every intentional change
from the legacy logic is a switch on :class:`ModernConfig`, so the pipeline can be
run in *compat* mode (all switches off, must equal the legacy output exactly) and
in *target* mode (all switches on), and each change can be attributed separately.
All arithmetic is in integers (cents, micro-cents, basis points), so results are
bit-identical across engines.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa

from .config import ELR_BP, ELR_OTHER_BP, IBNR_FROM_QUARTER, LOBS, QUARTERS
from .legacy import excel_vlookup_approx


@dataclass(frozen=True)
class ModernConfig:
    """Intentional differences D1-D6 (see results/rule_inventory.md). False = legacy behaviour."""

    normalise_lob: bool = False  # D4  trim + upper-case LOB codes before mapping
    asof_version: bool = False  # D2  policy attributes as at quarter end, not first record in file
    coalesce_reserve: bool = False  # D3  missing case reserve = 0, so paid is not dropped from incurred
    exact_earning: bool = False  # D1  earn in 1/10,000 cent, round once per report cell
    explicit_other_elr: bool = False  # D5  exact ELR lookup with an explicit OTHER row
    null_lr_zero_earned: bool = False  # D6  loss ratio NULL (not 0) when earned premium is 0

    @classmethod
    def target(cls) -> ModernConfig:
        return cls(**{f.name: True for f in fields(cls)})


COMPAT = ModernConfig()
TARGET = ModernConfig.target()

# Order in which differences are switched on for step-wise attribution.
DIFFERENCE_ORDER = [
    ("D4", "normalise_lob"),
    ("D2", "asof_version"),
    ("D3", "coalesce_reserve"),
    ("D1", "exact_earning"),
    ("D5", "explicit_other_elr"),
    ("D6", "null_lr_zero_earned"),
]


def attribution_steps() -> list[tuple[str, ModernConfig]]:
    """[("COMPAT", cfg0), ("D4", cfg1), ...] cumulative configs."""
    out = [("COMPAT", COMPAT)]
    cfg = COMPAT
    for did, flag in DIFFERENCE_ORDER:
        cfg = replace(cfg, **{flag: True})
        out.append((did, cfg))
    return out


# ----------------------------------------------------------------------------- helpers
def half_up_div(num: ir.IntegerValue, den: ir.IntegerValue | int) -> ir.IntegerValue:
    """round(num/den) half away from zero, for num >= 0, den > 0 (integer arithmetic)."""
    return (num * 2 + den) // (den * 2)


def half_away_div(num: ir.IntegerValue, den: ir.IntegerValue | int) -> ir.IntegerValue:
    """round(num/den) half away from zero for signed num, den > 0."""
    return ibis.ifelse(num >= 0, half_up_div(num, den), -half_up_div(-num, den))


def quarters_table(close_q: int) -> ir.Table:
    qs = QUARTERS[:close_q]
    return ibis.memtable(
        pa.table(
            {
                "quarter": pa.array([q.idx for q in qs], pa.int64()),
                "q_start": pa.array([q.start for q in qs], pa.int64()),
                "q_end": pa.array([q.end for q in qs], pa.int64()),
            }
        )
    )


def map_lob(raw: ir.StringValue, cfg: ModernConfig) -> ir.StringValue:
    code = raw.strip().upper() if cfg.normalise_lob else raw
    return ibis.ifelse(code.isin(LOBS), code, ibis.literal("OTHER"))


# ----------------------------------------------------------------------------- rules
def policy_attributes(pv: ir.Table, close_q: int, cfg: ModernConfig) -> ir.Table:
    """(policy_id, quarter) -> lob, region.  R01/R02 in the inventory."""
    q = quarters_table(close_q)
    if cfg.asof_version:
        cand = pv.cross_join(q).filter(lambda t: t.version_eff <= t.q_end)
        pick = cand.group_by(["policy_id", "quarter"]).aggregate(version_no=cand.version_no.max())
        chosen = pick.join(pv, ["policy_id", "version_no"])
        chosen = chosen.select("policy_id", "quarter", "lob_code_raw", "region")
        # quarters before the first version (no earning/claims can exist there) are not needed
    else:
        first = pv.group_by("policy_id").aggregate(load_seq=pv.load_seq.min())
        one = first.join(pv, ["policy_id", "load_seq"]).select("policy_id", "lob_code_raw", "region")
        chosen = one.cross_join(q.select("quarter"))
    return chosen.mutate(lob=map_lob(chosen.lob_code_raw, cfg)).select("policy_id", "quarter", "lob", "region")


def premium_detail(pt: ir.Table, attrs: ir.Table, close_q: int, cfg: ModernConfig) -> ir.Table:
    """Written and earned premium per transaction x quarter.  R03/R04/R05."""
    q = quarters_table(close_q)
    x = pt.cross_join(q)
    days = ibis.greatest(ibis.least(x.txn_exp, x.q_end + 1) - ibis.greatest(x.txn_eff, x.q_start), 0)
    written = ibis.ifelse((x.txn_eff >= x.q_start) & (x.txn_eff <= x.q_end), x.premium_cents, 0)
    term = x.txn_exp - x.txn_eff
    if cfg.exact_earning:
        earned_u = half_up_div(x.premium_cents * days * 10_000, term)
    else:
        earned_u = half_up_div(x.premium_cents * days, term) * 10_000
    x = x.mutate(days=days, written_cents=written.cast("int64"), earned_ucents=earned_u.cast("int64"))
    x = x.filter((x.days > 0) | (x.written_cents != 0))
    x = x.join(attrs, ["policy_id", "quarter"])
    return x.select("txn_id", "quarter", "policy_id", "lob", "region", "written_cents", "earned_ucents")


def claims_detail(cv: ir.Table, attrs: ir.Table, cfg: ModernConfig) -> ir.Table:
    """Paid, reserve and incurred movements per claim x valuation quarter.  R06/R07."""
    reserve = cv.case_reserve_cents.fill_null(0) if cfg.coalesce_reserve else cv.case_reserve_cents
    cur = cv.mutate(quarter=cv.valuation_q, reserve=reserve, incurred_cum=cv.cum_paid_cents + reserve)
    prev = cur.select(claim_id="claim_id", pq=cur.quarter + 1, prev_paid="cum_paid_cents", prev_inc="incurred_cum", has_prev=ibis.literal(1))
    j = cur.left_join(prev, [cur.claim_id == prev.claim_id, cur.quarter == prev.pq])
    first = j.has_prev.isnull()
    paid_q = j.cum_paid_cents - ibis.ifelse(first, 0, j.prev_paid)
    # SAS: incurred_q = incurred_cum - prev_inc; missing if either is missing; prev_inc = 0 on FIRST.claim_id
    inc_q = j.incurred_cum - ibis.ifelse(first, ibis.literal(0, "int64"), j.prev_inc)
    j = j.mutate(paid_cents=paid_q.cast("int64"), incurred_cents=inc_q.cast("int64"))
    j = j.select(claim_id=j.claim_id, quarter=j.quarter, policy_id=j.policy_id, paid_cents=j.paid_cents, case_reserve_cents=j.reserve.cast("int64"),
                 incurred_cum_cents=j.incurred_cum.cast("int64"), incurred_cents=j.incurred_cents)  # fmt: skip
    j = j.join(attrs, ["policy_id", "quarter"])
    return j.select("claim_id", "quarter", "policy_id", "lob", "region", "paid_cents", "case_reserve_cents", "incurred_cum_cents", "incurred_cents")


def elr_bp(lob: ir.StringValue, cfg: ModernConfig) -> ir.IntegerValue:
    domain = [*LOBS, "OTHER"]
    if cfg.explicit_other_elr:
        mapping = {**ELR_BP, "OTHER": ELR_OTHER_BP}
    else:
        mapping = {k: excel_vlookup_approx(k, ELR_BP) for k in domain}  # VLOOKUP(..., TRUE) semantics
    return lob.substitute(mapping, else_=ibis.null("int64")).cast("int64")


def quarterly_report(prem: ir.Table, clm: ir.Table, cfg: ModernConfig) -> ir.Table:
    """Regulatory quarterly return by quarter x LOB x region.  R08-R11."""
    keys = ["quarter", "lob", "region"]
    p = prem.group_by(keys).aggregate(written_premium_cents=prem.written_cents.sum(), earned_u=prem.earned_ucents.sum())
    pif = prem.filter(prem.earned_ucents > 0)
    n = pif.group_by(keys).aggregate(policies_in_force=pif.policy_id.nunique())
    c = clm.group_by(keys).aggregate(
        claims_paid_cents=clm.paid_cents.sum(), case_reserve_cents=clm.case_reserve_cents.sum(), incurred_cents=clm.incurred_cents.sum()
    )
    cells = prem.select(keys).union(clm.select(keys)).distinct()
    r = cells.left_join(p, keys).select(cells, "written_premium_cents", "earned_u")
    r = r.left_join(c, keys).select(r, "claims_paid_cents", "case_reserve_cents", "incurred_cents")
    r = r.left_join(n, keys).select(r, "policies_in_force")
    r = r.mutate(
        written_premium_cents=r.written_premium_cents.fill_null(0).cast("int64"),
        earned_premium_cents=half_up_div(r.earned_u.fill_null(0), 10_000).cast("int64"),
        claims_paid_cents=r.claims_paid_cents.fill_null(0).cast("int64"),
        case_reserve_cents=r.case_reserve_cents.fill_null(0).cast("int64"),
        incurred_cents=r.incurred_cents.fill_null(0).cast("int64"),
        policies_in_force=r.policies_in_force.fill_null(0).cast("int64"),
    )
    lr = half_away_div(r.incurred_cents * 10_000, r.earned_premium_cents)
    zero_lr = ibis.null("int64") if cfg.null_lr_zero_earned else ibis.literal(0, "int64")
    r = r.mutate(loss_ratio_bp=ibis.ifelse(r.earned_premium_cents > 0, lr, zero_lr).cast("int64"))
    ibnr = ibis.greatest(half_up_div(r.earned_premium_cents * elr_bp(r.lob, cfg), 10_000) - r.incurred_cents, 0)
    r = r.mutate(ibnr_cents=ibis.ifelse(r.quarter >= IBNR_FROM_QUARTER, ibnr, ibis.null("int64")).cast("int64"))
    labels = {q.idx: q.label for q in QUARTERS}
    r = r.mutate(quarter=r.quarter.substitute(labels, else_=ibis.null("string")))
    return r.select(
        "quarter", "lob", "region", "written_premium_cents", "earned_premium_cents", "claims_paid_cents", "case_reserve_cents",
        "incurred_cents", "loss_ratio_bp", "ibnr_cents", "policies_in_force",
    )  # fmt: skip


# ----------------------------------------------------------------------------- execution
REPORT_KEYS = ["quarter", "lob", "region"]
PREM_KEYS = ["txn_id", "quarter"]
CLAIM_KEYS = ["claim_id", "quarter"]


def connect(backend: str):
    if backend == "duckdb":
        return ibis.duckdb.connect()
    if backend == "polars":
        return ibis.polars.connect()
    raise ValueError(backend)


def _to_pandas(con, expr: ir.Table, keys: list[str]) -> pd.DataFrame:
    df = con.to_pyarrow(expr).to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    for c in df.columns:
        if str(df[c].dtype) not in ("Int64",):
            df[c] = df[c].astype("string").astype(object)
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


def build(sources: dict[str, pa.Table], close_q: int, cfg: ModernConfig) -> dict[str, ir.Table]:
    """Build the (backend-agnostic) expressions for the three outputs."""
    pv = ibis.memtable(sources["policy_versions"])
    pt = ibis.memtable(sources["premium_txns"])
    cv = ibis.memtable(sources["claim_valuations"])
    attrs = policy_attributes(pv, close_q, cfg)
    prem = premium_detail(pt, attrs, close_q, cfg)
    clm = claims_detail(cv, attrs, cfg)
    return {"premium_detail": prem, "claims_detail": clm, "report": quarterly_report(prem, clm, cfg)}


def run_modern_arrow(sources: dict[str, pa.Table], close_q: int, cfg: ModernConfig, backend: str = "duckdb", con=None) -> dict[str, pa.Table]:
    """Execute to Arrow (the form written to Iceberg); used by the benchmark."""
    con = con or connect(backend)
    return {k: con.to_pyarrow(e) for k, e in build(sources, close_q, cfg).items()}


def run_modern(sources: dict[str, pa.Table], close_q: int, cfg: ModernConfig, backend: str = "duckdb", con=None) -> dict[str, pd.DataFrame]:
    """Build the expressions once and execute them on the chosen backend; return sorted pandas frames."""
    con = con or connect(backend)
    e = build(sources, close_q, cfg)
    prem, clm, rep = e["premium_detail"], e["claims_detail"], e["report"]
    return {
        "premium_detail": _to_pandas(con, prem, PREM_KEYS),
        "claims_detail": _to_pandas(con, clm, CLAIM_KEYS),
        "report": _to_pandas(con, rep, REPORT_KEYS),
    }
