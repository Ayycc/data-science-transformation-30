"""Production monitoring with Evidently: monthly data-drift + regression-performance reports and the
retraining trigger policy."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from claims_mlops import config as C

PRED = "prediction"


def _definition(with_regression: bool) -> DataDefinition:
    nums = list(C.NUM_FEATURES)
    if with_regression:
        nums += [C.TARGET, PRED]
    return DataDefinition(
        numerical_columns=nums,
        categorical_columns=list(C.CAT_FEATURES),
        regression=[Regression(target=C.TARGET, prediction=PRED)] if with_regression else None,
    )


@dataclass
class DriftResult:
    month: int
    drift_share: float
    drifted_columns: list[str]
    column_scores: dict[str, float]
    column_methods: dict[str, str]


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, month: int, html_path: Path | None = None, max_rows: int = 25_000, seed: int = C.SEED
) -> DriftResult:
    """Evidently DataDriftPreset over the model features (+ RegressionPreset when predictions exist).

    Column-level drift flags come from Evidently's own ValueDrift tests (default method per column type
    and sample size: normed Wasserstein for numerics and Jensen-Shannon for categoricals above 1,000 rows).
    """
    ref = reference.sample(n=min(max_rows, len(reference)), random_state=seed)
    cur = current.sample(n=min(max_rows, len(current)), random_state=seed)
    has_pred = PRED in ref.columns and PRED in cur.columns and C.TARGET in cur.columns
    cols = C.FEATURES + ([C.TARGET, PRED] if has_pred else [])
    dd = _definition(has_pred)
    metrics = [DataDriftPreset(columns=list(C.FEATURES))]
    if has_pred and html_path is not None:
        metrics.append(RegressionPreset())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = Report(metrics, include_tests=True).run(
            Dataset.from_pandas(cur[cols].reset_index(drop=True), data_definition=dd),
            Dataset.from_pandas(ref[cols].reset_index(drop=True), data_definition=dd),
        )
        d = snap.dict()
        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            snap.save_html(str(html_path))
    by_id = {m["id"]: m for m in d["metrics"]}
    scores, methods, drifted = {}, {}, []
    for t in d.get("tests", []):
        cfg = t.get("metric_config", {}).get("params", {})
        if cfg.get("type", "").endswith("ValueDrift") and cfg.get("column") in C.FEATURES:
            col = cfg["column"]
            scores[col] = float(by_id[t["metric_config"]["metric_id"]]["value"])
            methods[col] = cfg.get("method", "")
            if t["status"] == "FAIL":
                drifted.append(col)
    drifted = [c for c in C.FEATURES if c in drifted]
    return DriftResult(month, len(drifted) / len(C.FEATURES), drifted, scores, methods)


@dataclass
class TriggerDecision:
    month: int
    fire: bool
    reasons: list[str]
    suppressed_by_cooldown: bool


def retrain_trigger(
    month: int, drift: DriftResult, perf: dict, ref_deviance: float, months_since_promotion: int, policy: C.TriggerPolicy
) -> TriggerDecision:
    """Fire when covariate drift OR performance degradation crosses policy thresholds (outside cooldown)."""
    reasons = []
    if drift.drift_share >= policy.drift_share:
        reasons.append(f"drift_share {drift.drift_share:.2f} >= {policy.drift_share:.2f} ({', '.join(drift.drifted_columns)})")
    if abs(perf["ae_ratio"] - 1) > policy.ae_tolerance:
        reasons.append(f"|A/E-1| {abs(perf['ae_ratio'] - 1):.3f} > {policy.ae_tolerance:.2f}")
    ratio = perf["tweedie_deviance"] / ref_deviance
    if ratio > policy.deviance_ratio:
        reasons.append(f"deviance ratio {ratio:.3f} > {policy.deviance_ratio:.2f}")
    cooling = months_since_promotion < policy.cooldown_months
    return TriggerDecision(month, bool(reasons) and not cooling, reasons, bool(reasons) and cooling)


def to_dict(obj) -> dict:
    out = asdict(obj)
    for k, v in out.items():
        if isinstance(v, float | np.floating):
            out[k] = float(v)
    return out
