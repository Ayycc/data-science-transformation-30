"""Inventory of the legacy business rules, their dependency graph, the documented
intentional differences, and column-level lineage from legacy rule to modern model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rid: str
    name: str
    statement: str
    legacy_source: str
    modern_impl: str
    depends_on: tuple[str, ...]
    differences: tuple[str, ...] = ()


RULES: list[Rule] = [
    Rule("R01", "Policy attributes", "Each policy contributes its LOB and region from one policy version.",
         "sas/01_policy_prep.sas: PROC SORT NODUPKEY BY policy_id", "modern.policy_attributes", (), ("D2",)),
    Rule("R02", "LOB mapping", "Raw LOB codes map to the five reporting LOBs; anything else is OTHER.",
         "sas/01_policy_prep.sas: DATA pol_dedup IF lob_code_raw IN (...)", "modern.map_lob", ("R01",), ("D4",)),
    Rule("R03", "Written premium", "Premium is written in the quarter containing the transaction effective date.",
         "sas/02_premium_earn.sas: IF qs <= txn_eff <= qe", "modern.premium_detail", ("R01", "R02")),
    Rule("R04", "Earned premium", "Premium earns daily pro rata over [txn_eff, txn_exp).",
         "sas/02_premium_earn.sas: earned_cents = ROUND(premium*days/term, 1)", "modern.premium_detail", ("R01", "R02"), ("D1",)),
    Rule("R05", "Policies in force", "Distinct policies with earned premium > 0 in the quarter.",
         "sas/04_summarise.sas: PROC SORT NODUPKEY WHERE earned > 0", "modern.quarterly_report", ("R04",)),
    Rule("R06", "Paid movement", "Quarterly paid = cumulative paid minus previous valuation's cumulative paid.",
         "sas/03_claims.sas: RETAIN prev_paid; FIRST.claim_id", "modern.claims_detail", ("R01", "R02")),
    Rule("R07", "Incurred movement", "Incurred = paid + case reserve; quarterly incurred is its movement.",
         "sas/03_claims.sas: incurred_cum = cum_paid + case_reserve", "modern.claims_detail", ("R06",), ("D3",)),
    Rule("R08", "Return aggregation", "Sum all measures by quarter x LOB x region (outer merge of premium and claims).",
         "sas/04_summarise.sas: PROC SUMMARY NWAY; MERGE", "modern.quarterly_report", ("R03", "R04", "R05", "R06", "R07")),
    Rule("R09", "Loss ratio", "Loss ratio = incurred / earned to 4 dp (basis points).",
         "excel/QRT_build.bas: =IFERROR(ROUND(H2/E2,4),0)", "modern.quarterly_report", ("R08", "R11"), ("D6",)),
    Rule("R10", "IBNR", "IBNR = max(0, earned x expected loss ratio - incurred), from 2025Q1.",
         "excel/QRT_build.bas: =MAX(0,ROUND(E2*VLOOKUP(B2,ELR,2,TRUE),0)-H2)", "modern.quarterly_report + modern.elr_bp", ("R08", "R11"), ("D5",)),
    Rule("R11", "Rounding", "Money is reported in whole cents, rounded half away from zero.",
         "sas/02_premium_earn.sas ROUND(...,1); excel ROUND(...,0)", "modern.half_up_div", ("R04",), ("D1",)),
]  # fmt: skip


@dataclass(frozen=True)
class Difference:
    did: str
    title: str
    legacy: str
    modern: str
    why: str
    predicate: str


DIFFERENCES: list[Difference] = [
    Difference("D1", "Single rounding instead of row-level rounding",
               "Earned premium rounded to the cent on every txn x quarter row, then summed.",
               "Earned in 1/10,000 cent per row; rounded to the cent once per report cell.",
               "Row-level rounding accumulates up to 0.5 cent per row; the regulator's rule is to round the reported figure.",
               "Detail: only earned_ucents changes, by <= 0.5 cent. Report: earned changes by <= 0.5 cent x rows in cell + 1; "
               "loss_ratio_bp / ibnr_cents may change only where earned changed."),
    Difference("D2", "Policy attributes as at quarter end",
               "First policy version in physical file order (NODUPKEY): stale after endorsements, wrong after the 2025 backfill.",
               "Latest version effective on or before each quarter end.",
               "Region moves after endorsements must follow the policy; the result must not depend on file order.",
               "Detail: only lob/region change, only for (policy, quarter) pairs whose legacy version differs from the as-of version. "
               "Report: only cells that contain such a pair."),
    Difference("D3", "Missing case reserve treated as zero",
               "SAS '+' makes incurred missing when the reserve is missing; PROC SUMMARY then drops it (and the next quarter's movement).",
               "Missing reserve coalesced to 0, so paid stays in incurred.",
               "Paid claims must never disappear from incurred because a reserve feed failed.",
               "Detail: only reserve/incurred columns, only claim-quarters with a missing reserve at q or q-1. "
               "Report: only incurred/loss ratio/IBNR in cells containing such claim-quarters."),
    Difference("D4", "LOB codes trimmed and upper-cased",
               "Case- and blank-sensitive IN() list: 'Motor', 'home ' become OTHER.",
               "TRIM + UPPER before mapping; only genuinely unmapped codes (PET) are OTHER.",
               "Premium and claims were being reported under the wrong line of business.",
               "Detail: only lob changes, only for policies with a fixable raw code. Report: only cells containing those policies."),
    Difference("D5", "Exact ELR lookup with an explicit OTHER row",
               "VLOOKUP(...,TRUE) on an A-Z table: OTHER silently takes MOTOR's 71% ELR.",
               "Exact lookup; OTHER has its own 65% ELR row.",
               "Approximate match on text keys is a known Excel defect.",
               "Report only: ibnr_cents on lob = OTHER rows."),
    Difference("D6", "Loss ratio NULL when earned premium is zero",
               "IFERROR(...,0) reports a 0% loss ratio.",
               "NULL (not meaningful).",
               "A 0% loss ratio on a cell with claims but no premium is misleading.",
               "Report only: loss_ratio_bp goes from 0 to NULL where earned premium is 0."),
]  # fmt: skip

# Column-level lineage: report column -> (source columns, rules)
LINEAGE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "lob": (("policy_versions.lob_code_raw", "policy_versions.load_seq", "policy_versions.version_eff"), ("R01", "R02")),
    "region": (("policy_versions.region", "policy_versions.load_seq", "policy_versions.version_eff"), ("R01",)),
    "written_premium_cents": (("premium_txns.premium_cents", "premium_txns.txn_eff"), ("R03", "R08")),
    "earned_premium_cents": (("premium_txns.premium_cents", "premium_txns.txn_eff", "premium_txns.txn_exp"), ("R04", "R08", "R11")),
    "policies_in_force": (("premium_txns.policy_id",), ("R04", "R05", "R08")),
    "claims_paid_cents": (("claim_valuations.cum_paid_cents",), ("R06", "R08")),
    "case_reserve_cents": (("claim_valuations.case_reserve_cents",), ("R07", "R08")),
    "incurred_cents": (("claim_valuations.cum_paid_cents", "claim_valuations.case_reserve_cents"), ("R06", "R07", "R08")),
    "loss_ratio_bp": (("incurred_cents", "earned_premium_cents"), ("R09",)),
    "ibnr_cents": (("earned_premium_cents", "incurred_cents", "ELR table"), ("R10",)),
}


def rule_map() -> dict[str, Rule]:
    return {r.rid: r for r in RULES}


def topo_order() -> list[str]:
    """Topological order of the rules; raises on a cycle or a dangling dependency."""
    rm = rule_map()
    order, state = [], {}

    def visit(r: str) -> None:
        if r not in rm:
            raise KeyError(f"unknown rule {r}")
        if state.get(r) == 1:
            raise ValueError(f"cycle at {r}")
        if state.get(r) == 2:
            return
        state[r] = 1
        for d in rm[r].depends_on:
            visit(d)
        state[r] = 2
        order.append(r)

    for r in rm:
        visit(r)
    return order


def depth() -> dict[str, int]:
    rm, out = rule_map(), {}
    for r in topo_order():
        out[r] = 1 + max((out[d] for d in rm[r].depends_on), default=-1)
    return out


def mermaid_graph() -> str:
    lines = ["flowchart LR"]
    for r in RULES:
        tag = f"<br/>{', '.join(r.differences)}" if r.differences else ""
        lines.append(f'    {r.rid}["{r.rid} {r.name}{tag}"]')
    for r in RULES:
        for d in r.depends_on:
            lines.append(f"    {d} --> {r.rid}")
    diff_nodes = sorted({r.rid for r in RULES if r.differences})
    lines.append("    classDef diff fill:#fde2c8,stroke:#c2570c")
    lines.append(f"    class {','.join(diff_nodes)} diff")
    return "\n".join(lines)


def inventory_markdown() -> str:
    out = ["# Legacy rule inventory", "", "Generated by `make run` from `src/legacy28/rules.py`.", "",
           "| Rule | Name | Business rule | Legacy source | Modern implementation | Depends on | Intentional differences |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in RULES:
        out.append(
            f"| {r.rid} | {r.name} | {r.statement} | `{r.legacy_source}` | `{r.modern_impl}` | {', '.join(r.depends_on) or '-'} | {', '.join(r.differences) or '-'} |"
        )
    out += ["", "## Dependency graph", "", "```mermaid", mermaid_graph(), "```", "", "## Intentional differences", ""]
    out += ["| ID | Change | Legacy behaviour | Modern behaviour | Why | Reconciliation predicate (what may change) |", "|---|---|---|---|---|---|"]
    for d in DIFFERENCES:
        out.append(f"| {d.did} | {d.title} | {d.legacy} | {d.modern} | {d.why} | {d.predicate} |")
    return "\n".join(out) + "\n"


def lineage_markdown() -> str:
    rm = rule_map()
    out = ["# Column-level lineage: legacy rule -> modern model", "", "Generated by `make run` from `src/legacy28/rules.py`.", "",
           "| Report column | Source columns | Rules | Legacy code | Modern code | Differences |", "|---|---|---|---|---|---|"]  # fmt: skip
    for col, (srcs, rules) in LINEAGE.items():
        legacy = "<br/>".join(f"`{rm[r].legacy_source}`" for r in rules)
        modern = ", ".join(sorted({f"`{rm[r].modern_impl}`" for r in rules}))
        diffs = ", ".join(sorted({d for r in rules for d in rm[r].differences})) or "-"
        out.append(f"| `{col}` | {', '.join(f'`{s}`' for s in srcs)} | {', '.join(rules)} | {legacy} | {modern} | {diffs} |")
    out += ["", "```mermaid", "flowchart LR"]
    for col, (_srcs, rules) in LINEAGE.items():
        for r in rules:
            out.append(f"    {r} --> {col}")
    out.append("```")
    return "\n".join(out) + "\n"
