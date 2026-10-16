"""Evaluation against the generator's ground truth (possible only because the data is SYNTHETIC)."""

from __future__ import annotations

import pandas as pd

from .contract import RULES

EXPECTED_RULES: dict[str, set[str]] = {
    "COORDS_MISSING": {"COORDS_MISSING"},
    "COORDS_NULL_ISLAND": {"COORDS_NULL_ISLAND"},
    "COORDS_SWAPPED": {"COORDS_SWAPPED_LATLON"},
    "COORDS_OUTSIDE_CITY": {"COORDS_OUTSIDE_CITY", "COORDS_OUT_OF_BBOX"},
    "CLOSED_BEFORE_CREATED": {"CLOSED_BEFORE_CREATED"},
    "CREATED_IN_FUTURE": {"CREATED_IN_FUTURE"},
    "CLOSED_PLACEHOLDER": {"CLOSED_PLACEHOLDER_DATE"},
    "TIMESTAMP_UNPARSEABLE": {"CREATED_MISSING_OR_UNPARSEABLE"},
    "COMPLETED_WITHOUT_CLOSE": {"COMPLETED_WITHOUT_CLOSE_DATE"},
    "OPEN_WITH_CLOSE": {"OPEN_WITH_CLOSE_DATE"},
    "STATUS_INVALID": {"STATUS_INVALID"},
    "WARD_INVALID": {"WARD_INVALID"},
    "SR_NUMBER_MALFORMED": {"SR_NUMBER_MALFORMED", "SR_NUMBER_MISSING"},
    "CATEGORY_UNMAPPABLE": {"CATEGORY_UNMAPPED"},
    "DUPLICATE_RECORD": {"DUPLICATE_RECORD"},
}


def defect_detection(truth: pd.DataFrame, rejects: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per injected defect class: rows, rows caught by an expected rule, rows quarantined by any rule."""
    codes = rejects.groupby("source_line").reason_code.agg(set)
    t = truth.assign(source_line=range(1, len(truth) + 1))
    t["codes"] = t.source_line.map(codes)
    rows = []
    for cls, exp in EXPECTED_RULES.items():
        sub = t[t.defect == cls]
        caught = sub.codes.apply(lambda c, e=exp: isinstance(c, set) and bool(c & e))
        anyq = sub.codes.apply(lambda c: isinstance(c, set))
        fired = sorted(set().union(*[c for c in sub.codes if isinstance(c, set)])) if len(sub) else []
        rows.append(
            {
                "defect_class": cls, "rows": len(sub), "caught_by_expected_rule": int(caught.sum()),
                "recall": round(float(caught.mean()), 4) if len(sub) else None,
                "quarantined_any_rule": int(anyq.sum()), "expected_rules": "|".join(sorted(exp)),
                "rules_fired": "|".join(fired),
            }
        )  # fmt: skip
    clean = t[t.defect == ""]
    fp = clean.codes.apply(lambda c: isinstance(c, set))
    summary = {
        "defect_classes": len(EXPECTED_RULES),
        "classes_fully_caught": int(sum(r["rows"] > 0 and r["caught_by_expected_rule"] == r["rows"] for r in rows)),
        "classes_present": int(sum(r["rows"] > 0 for r in rows)),
        "defect_rows": int((t.defect != "").sum()),
        "defect_rows_caught": int(sum(r["caught_by_expected_rule"] for r in rows)),
        "clean_rows": int(len(clean)),
        "clean_rows_quarantined": int(fp.sum()),
        "rules_never_fired": [r.code for r in RULES if r.code not in set(rejects.reason_code)],
    }
    return pd.DataFrame(rows), summary


def harmonisation_accuracy(staged: pd.DataFrame, truth: pd.DataFrame, auto_only: pd.Series) -> dict:
    """Row-level accuracy of category harmonisation, with and without the reviewed mapping table."""
    t = truth.reset_index(drop=True)
    junk = t.variant_kind == "junk"
    real = ~junk
    variant = real & (t.variant_kind != "canonical")
    final = staged.category.reset_index(drop=True)
    auto = auto_only.reset_index(drop=True)

    def acc(pred: pd.Series, mask: pd.Series) -> dict:
        p, y = pred[mask], t.true_category[mask]
        return {
            "rows": int(mask.sum()), "correct": int((p == y).sum()), "wrong": int((p.notna() & (p != y)).sum()),
            "unresolved": int(p.isna().sum()), "accuracy": round(float((p == y).mean()), 6),
        }  # fmt: skip

    by_kind = {}
    for k in sorted(t.loc[real, "variant_kind"].unique()):
        m = t.variant_kind == k
        by_kind[k] = {"final": acc(final, m), "auto_only": acc(auto, m)}
    return {
        "all_real_rows": {"final": acc(final, real), "auto_only": acc(auto, real)},
        "variant_rows": {"final": acc(final, variant), "auto_only": acc(auto, variant)},
        "by_variant_kind": by_kind,
        "junk_rows": int(junk.sum()),
        "junk_rows_mapped_to_a_category": int(final[junk].notna().sum()),
    }
