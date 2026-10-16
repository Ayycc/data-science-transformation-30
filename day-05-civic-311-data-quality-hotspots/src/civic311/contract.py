"""The data contract for the 311 extract: one rule registry, two implementations (pandera and Great Expectations).

Each rule has a stable id and a reason code. pandera is the enforcement layer (its failure cases become the
quarantine table, so every rejected row points at a rule id); Great Expectations is the independent audit layer
that produces the validation report for the raw batch and re-certifies the clean batch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

from .config import EXTRACT_TS, N_WARDS, PLACEHOLDER_CUTOFF, WINDOW_START
from .geo import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

CONTRACT_VERSION = "311-sr-contract/1.0.0"
STATUSES = ("Open", "Completed", "Canceled")
SR_PATTERN = r"^SR\d{2}-\d{8}$"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    code: str
    column: str
    description: str
    ge: str  # the Great Expectations expectation that implements it
    ge_comparable: bool  # the sum of GE unexpected_counts for the rule has exactly the pandera row semantics


RULES: tuple[Rule, ...] = (
    Rule("R01", "SR_NUMBER_MISSING", "sr_number", "Request id is present", "expect_column_values_to_not_be_null", True),
    Rule("R02", "SR_NUMBER_MALFORMED", "sr_number", "Request id matches ^SR\\d{2}-\\d{8}$",
         "expect_column_values_to_match_regex", True),
    Rule("R03", "DUPLICATE_RECORD", "sr_number", "Request id not seen earlier in the extract (first copy kept)",
         "expect_column_values_to_be_unique", False),
    Rule("R04", "CREATED_MISSING_OR_UNPARSEABLE", "created_ts", "Created timestamp present and parseable",
         "expect_column_values_to_not_be_null", True),
    Rule("R05", "CREATED_IN_FUTURE", "created_ts", "Created at or before the extract time",
         "expect_column_values_to_be_between", True),
    Rule("R06", "CREATED_BEFORE_WINDOW", "created_ts", "Created on or after the extract window start",
         "expect_column_values_to_be_between", True),
    Rule("R07", "CLOSED_UNPARSEABLE", "closed_ts", "Closed timestamp parseable when present",
         "expect_column_values_to_not_be_null (row_condition: closed_raw present)", True),
    Rule("R08", "CLOSED_PLACEHOLDER_DATE", "closed_ts", "Closed date is not a sentinel (before 2000-01-01)",
         "expect_column_values_to_be_between", True),
    Rule("R09", "CLOSED_IN_FUTURE", "closed_ts", "Closed at or before the extract time",
         "expect_column_values_to_be_between", True),
    Rule("R10", "CLOSED_BEFORE_CREATED", "closed_ts", "Closed at or after created",
         "expect_column_pair_values_a_to_be_greater_than_b", True),
    Rule("R11", "STATUS_INVALID", "status", "Status in {Open, Completed, Canceled}",
         "expect_column_values_to_be_in_set + not_be_null", True),
    Rule("R12", "COMPLETED_WITHOUT_CLOSE_DATE", "closed_raw", "Completed requests carry a closed date",
         "expect_column_values_to_not_be_null (row_condition: status == Completed)", True),
    Rule("R13", "OPEN_WITH_CLOSE_DATE", "closed_raw", "Open requests carry no closed date",
         "expect_column_values_to_be_null (row_condition: status == Open)", True),
    Rule("R14", "COORDS_MISSING", "latitude", "Latitude and longitude both present and numeric",
         "expect_column_values_to_not_be_null (x2)", False),
    Rule("R15", "COORDS_NULL_ISLAND", "latitude", "Coordinates are not (0, 0)",
         "expect_column_values_to_not_be_in_set (row_condition: longitude == 0)", True),
    Rule("R16", "COORDS_OUT_OF_BBOX", "latitude", "Coordinates inside the city bounding box",
         "expect_column_values_to_be_between (x2)", False),
    Rule("R17", "COORDS_SWAPPED_LATLON", "latitude", "Coordinates are not lat/lon swapped (swapped pair lands in bbox)",
         "expect_column_values_to_be_in_set (swapped_in_bbox == False, row_condition: outside bbox)", True),
    Rule("R18", "COORDS_OUTSIDE_CITY", "latitude", "Coordinates inside the city polygon (when inside the bbox)",
         "expect_column_values_to_be_in_set (in_city_polygon == True, row_condition: in bbox)", True),
    Rule("R19", "WARD_INVALID", "ward", f"Ward present, integer, 1..{N_WARDS}",
         "expect_column_values_to_be_between + not_be_null", True),
    Rule("R20", "CATEGORY_UNMAPPED", "category", "Raw category maps to a canonical category (auto or reviewed)",
         "expect_column_values_to_be_in_set + not_be_null", True),
)  # fmt: skip
BY_CODE = {r.code: r for r in RULES}
BY_ID = {r.rule_id: r for r in RULES}

# pandera reports a nullable=False failure as check "not_nullable" on the column; map it to a reason code.
NOT_NULL_CODE = {
    "sr_number": "SR_NUMBER_MISSING",
    "created_ts": "CREATED_MISSING_OR_UNPARSEABLE",
    "status": "STATUS_INVALID",
    "latitude": "COORDS_MISSING",
    "longitude": "COORDS_MISSING",
    "ward": "WARD_INVALID",
    "category": "CATEGORY_UNMAPPED",
}

_EXTRACT = pd.Timestamp(EXTRACT_TS)
_START = pd.Timestamp(WINDOW_START)
_PLACEHOLDER = pd.Timestamp(PLACEHOLDER_CUTOFF)


def _ok(mask: pd.Series) -> pd.Series:
    """pandera row-wise checks return True for passing rows."""
    return ~mask.fillna(False).astype(bool)


def _coords(df: pd.DataFrame) -> pd.Series:
    return df.latitude.notna() & df.longitude.notna()


def _null_island(df: pd.DataFrame) -> pd.Series:
    return (df.latitude == 0) & (df.longitude == 0)


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Plain-pandas re-derivation of every rule (True = row violates). Used to audit the quarantine table."""
    xy = _coords(df)
    m = {
        "SR_NUMBER_MISSING": df.sr_number.isna(),
        "SR_NUMBER_MALFORMED": df.sr_number.notna() & ~df.sr_number.fillna("").str.match(SR_PATTERN),
        "DUPLICATE_RECORD": df.sr_number.notna() & df.sr_number.duplicated(keep="first"),
        "CREATED_MISSING_OR_UNPARSEABLE": df.created_ts.isna(),
        "CREATED_IN_FUTURE": df.created_ts > _EXTRACT,
        "CREATED_BEFORE_WINDOW": df.created_ts < _START,
        "CLOSED_UNPARSEABLE": df.closed_raw.notna() & df.closed_ts.isna(),
        "CLOSED_PLACEHOLDER_DATE": df.closed_ts < _PLACEHOLDER,
        "CLOSED_IN_FUTURE": df.closed_ts > _EXTRACT,
        "CLOSED_BEFORE_CREATED": df.closed_ts < df.created_ts,
        "STATUS_INVALID": ~df.status.isin(STATUSES),
        "COMPLETED_WITHOUT_CLOSE_DATE": (df.status == "Completed") & df.closed_raw.isna(),
        "OPEN_WITH_CLOSE_DATE": (df.status == "Open") & df.closed_raw.notna(),
        "COORDS_MISSING": ~xy,
        "COORDS_NULL_ISLAND": _null_island(df),
        "COORDS_OUT_OF_BBOX": xy & ~df.in_bbox & ~_null_island(df),
        "COORDS_SWAPPED_LATLON": xy & ~df.in_bbox & df.swapped_in_bbox,
        "COORDS_OUTSIDE_CITY": xy & df.in_bbox & ~df.in_city_polygon,
        "WARD_INVALID": df.ward.isna() | (df.ward < 1) | (df.ward > N_WARDS),
        "CATEGORY_UNMAPPED": df.category.isna(),
    }
    return {k: v.fillna(False).astype(bool) for k, v in m.items()}


def _named(factory, *args, code: str) -> pa.Check:
    """A pandera check whose name and error string are the reason code (pandera reports the error string)."""
    return factory(*args, name=code, error=code)


def pandera_schema() -> pa.DataFrameSchema:
    C = pa.Check
    row_rules = {
        "CLOSED_UNPARSEABLE": lambda d: d.closed_raw.notna() & d.closed_ts.isna(),
        "CLOSED_BEFORE_CREATED": lambda d: d.closed_ts < d.created_ts,
        "COMPLETED_WITHOUT_CLOSE_DATE": lambda d: (d.status == "Completed") & d.closed_raw.isna(),
        "OPEN_WITH_CLOSE_DATE": lambda d: (d.status == "Open") & d.closed_raw.notna(),
        "COORDS_NULL_ISLAND": _null_island,
        "COORDS_OUT_OF_BBOX": lambda d: _coords(d) & ~d.in_bbox & ~_null_island(d),
        "COORDS_SWAPPED_LATLON": lambda d: _coords(d) & ~d.in_bbox & d.swapped_in_bbox,
        "COORDS_OUTSIDE_CITY": lambda d: _coords(d) & d.in_bbox & ~d.in_city_polygon,
        "DUPLICATE_RECORD": lambda d: d.sr_number.notna() & d.sr_number.duplicated(keep="first"),
    }
    row_checks = [C(lambda d, f=f: _ok(f(d)), name=code, error=code) for code, f in row_rules.items()]
    ts = "datetime64[ns]"
    return pa.DataFrameSchema(
        {
            "source_line": pa.Column("int64", unique=True, nullable=False),
            "sr_number": pa.Column(str, nullable=False,
                                   checks=_named(C.str_matches, SR_PATTERN, code="SR_NUMBER_MALFORMED")),
            "created_ts": pa.Column(ts, nullable=False, checks=[_named(C.le, _EXTRACT, code="CREATED_IN_FUTURE"),
                                                                _named(C.ge, _START, code="CREATED_BEFORE_WINDOW")]),
            "closed_ts": pa.Column(ts, nullable=True,
                                   checks=[_named(C.ge, _PLACEHOLDER, code="CLOSED_PLACEHOLDER_DATE"),
                                           _named(C.le, _EXTRACT, code="CLOSED_IN_FUTURE")]),
            "status": pa.Column(str, nullable=False, checks=_named(C.isin, STATUSES, code="STATUS_INVALID")),
            "latitude": pa.Column(float, nullable=False),
            "longitude": pa.Column(float, nullable=False),
            "ward": pa.Column("Int64", nullable=False, checks=_named(C.in_range, 1, N_WARDS, code="WARD_INVALID")),
            "category": pa.Column(str, nullable=False),
        },
        checks=row_checks,
        strict=False,
        name=CONTRACT_VERSION,
    )  # fmt: skip


def run_pandera(df: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily and return one row per (source_line, reason) with the failing value.

    Columns: source_line, sr_number, rule_id, reason_code, column, failure_value, detected_by, contract_version.
    """
    schema = pandera_schema()
    try:
        schema.validate(df, lazy=True)
        fc = pd.DataFrame(columns=["column", "check", "failure_case", "index"])
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
    fc = fc[fc["index"].notna()].copy()
    unknown = fc[~fc.check.astype(str).isin(set(BY_CODE) | {"not_nullable"})]
    if len(unknown):
        raise RuntimeError(f"unexpected pandera failures (schema/dtype problems):\n{unknown.head()}")
    fc["reason_code"] = [
        NOT_NULL_CODE[c] if chk == "not_nullable" else chk
        for c, chk in zip(fc.column, fc.check.astype(str), strict=True)
    ]
    fc["index"] = fc["index"].astype("int64")
    fc = fc.sort_values(["index", "reason_code", "column"]).drop_duplicates(["index", "reason_code"])
    rule = BY_CODE
    rows = df.loc[fc["index"].to_numpy()]
    raw_col = {"closed_ts": "closed_raw", "created_ts": "created_raw", "ward": "ward_raw", "category": "sr_type_raw"}
    failure_value = []
    for idx, code in zip(fc["index"], fc.reason_code, strict=True):
        if code.startswith("COORDS"):
            failure_value.append(f"{df.at[idx, 'latitude']},{df.at[idx, 'longitude']}")
        else:
            col = rule[code].column
            failure_value.append(str(df.at[idx, raw_col.get(col, col)]))
    return pd.DataFrame(
        {
            "source_line": rows.source_line.to_numpy(),
            "sr_number": rows.sr_number.to_numpy(),
            "rule_id": [rule[c].rule_id for c in fc.reason_code],
            "reason_code": fc.reason_code.to_numpy(),
            "column": [rule[c].column for c in fc.reason_code],
            "failure_value": failure_value,
            "detected_by": "pandera",
            "contract_version": CONTRACT_VERSION,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------- Great Expectations
def _quiet_gx() -> None:
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
    logging.getLogger("great_expectations").setLevel(logging.ERROR)


def ge_expectations() -> list[tuple[str, object]]:
    """(rule_id, expectation) pairs implementing the contract in Great Expectations."""
    _quiet_gx()
    import great_expectations as gx

    E = gx.expectations
    p = {"condition_parser": "pandas"}
    has_xy = "latitude.notna() and longitude.notna()"
    return [
        ("R01", E.ExpectColumnValuesToNotBeNull(column="sr_number")),
        ("R02", E.ExpectColumnValuesToMatchRegex(column="sr_number", regex=SR_PATTERN)),
        ("R03", E.ExpectColumnValuesToBeUnique(column="sr_number")),
        ("R04", E.ExpectColumnValuesToNotBeNull(column="created_ts")),
        ("R05", E.ExpectColumnValuesToBeBetween(column="created_ts", max_value=_EXTRACT)),
        ("R06", E.ExpectColumnValuesToBeBetween(column="created_ts", min_value=_START)),
        ("R07", E.ExpectColumnValuesToNotBeNull(column="closed_ts", row_condition="closed_raw.notna()", **p)),
        ("R08", E.ExpectColumnValuesToBeBetween(column="closed_ts", min_value=_PLACEHOLDER)),
        ("R09", E.ExpectColumnValuesToBeBetween(column="closed_ts", max_value=_EXTRACT)),
        ("R10", E.ExpectColumnPairValuesAToBeGreaterThanB(column_A="closed_ts", column_B="created_ts", or_equal=True,
                                                          ignore_row_if="either_value_is_missing")),
        ("R11", E.ExpectColumnValuesToNotBeNull(column="status")),
        ("R11", E.ExpectColumnValuesToBeInSet(column="status", value_set=list(STATUSES))),
        ("R12", E.ExpectColumnValuesToNotBeNull(column="closed_raw", row_condition='status == "Completed"', **p)),
        ("R13", E.ExpectColumnValuesToBeNull(column="closed_raw", row_condition='status == "Open"', **p)),
        ("R14", E.ExpectColumnValuesToNotBeNull(column="latitude")),
        ("R14", E.ExpectColumnValuesToNotBeNull(column="longitude")),
        ("R15", E.ExpectColumnValuesToNotBeInSet(column="latitude", value_set=[0.0], row_condition="longitude == 0",
                                                 **p)),
        ("R16", E.ExpectColumnValuesToBeBetween(column="latitude", min_value=LAT_MIN, max_value=LAT_MAX,
                                                row_condition="not (latitude == 0 and longitude == 0)", **p)),
        ("R16", E.ExpectColumnValuesToBeBetween(column="longitude", min_value=LON_MIN, max_value=LON_MAX,
                                                row_condition="not (latitude == 0 and longitude == 0)", **p)),
        ("R17", E.ExpectColumnValuesToBeInSet(column="swapped_in_bbox", value_set=[False],
                                              row_condition=f"{has_xy} and not in_bbox", **p)),
        ("R18", E.ExpectColumnValuesToBeInSet(column="in_city_polygon", value_set=[True],
                                              row_condition=f"{has_xy} and in_bbox", **p)),
        ("R19", E.ExpectColumnValuesToNotBeNull(column="ward")),
        ("R19", E.ExpectColumnValuesToBeBetween(column="ward", min_value=1, max_value=N_WARDS)),
        ("R20", E.ExpectColumnValuesToNotBeNull(column="category")),
    ]  # fmt: skip


def run_ge(df: pd.DataFrame, batch_name: str) -> dict:
    """Run the GE suite on a batch; return a compact JSON-able validation report."""
    _quiet_gx()
    import great_expectations as gx
    from great_expectations.data_context.types.base import ProgressBarsConfig

    ctx = gx.get_context(mode="ephemeral")
    ctx.variables.progress_bars = ProgressBarsConfig(globally=False)
    ds = ctx.data_sources.add_pandas("civic311")
    bd = ds.add_dataframe_asset(batch_name).add_batch_definition_whole_dataframe("whole")
    suite = ctx.suites.add(gx.ExpectationSuite(name=CONTRACT_VERSION))
    from .categories import CANONICAL

    pairs = [
        *ge_expectations(),
        ("R20", gx.expectations.ExpectColumnValuesToBeInSet(column="category", value_set=list(CANONICAL))),
    ]
    for rid, exp in pairs:
        exp.meta = {"rule_id": rid, "code": BY_ID[rid].code}
        suite.add_expectation(exp)
    gx_df = df.copy()
    gx_df["ward"] = gx_df["ward"].astype("float")  # GE's pandas engine does not handle nullable Int64 comparisons
    res = bd.get_batch(batch_parameters={"dataframe": gx_df}).validate(suite, result_format="SUMMARY")
    out = []
    for r in res.results:
        cfg = r.expectation_config
        rid = cfg.meta["rule_id"]
        kw = {k: v for k, v in cfg.kwargs.items() if k not in ("batch_id",)}
        out.append(
            {
                "rule_id": rid,
                "code": BY_ID[rid].code,
                "expectation": cfg.type,
                "kwargs": {k: str(v) for k, v in kw.items()},
                "success": bool(r.success),
                "element_count": int(r.result.get("element_count", 0) or 0),
                "unexpected_count": int(r.result.get("unexpected_count", 0) or 0),
                "unexpected_percent": float(r.result.get("unexpected_percent", 0.0) or 0.0),
                "partial_unexpected_list": [str(x) for x in (r.result.get("partial_unexpected_list") or [])[:5]],
                "exception": bool(r.exception_info.get("raised_exception", False)) if r.exception_info else False,
            }
        )
    out.sort(key=lambda d: (d["rule_id"], d["expectation"], d["kwargs"].get("column", "")))
    stats = res.statistics
    return {
        "batch": batch_name,
        "rows": len(df),
        "suite": CONTRACT_VERSION,
        "success": bool(res.success),
        "evaluated_expectations": int(stats["evaluated_expectations"]),
        "successful_expectations": int(stats["successful_expectations"]),
        "results": out,
    }
