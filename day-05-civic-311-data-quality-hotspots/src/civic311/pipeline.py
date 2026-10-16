"""End-to-end run: acquire -> stage (DuckDB) -> harmonise -> validate (pandera + GE) -> quarantine -> KPIs
-> anomalies -> H3 hotspots -> equity / SLO -> results/."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import categories, contract, equity, evaluate, generate, hotspots, ingest, kpi, report
from .config import (
    FDR_Q,
    H3_RES,
    MATERIAL_EXCESS,
    MIN_CELL_N,
    PUBLIC_PROBES,
    REVIEWED_MAPPING,
    SLO_PASS_TARGET,
    Scale,
)


def probe_public(timeout: float = 10.0) -> dict:
    """Try each public 311 API once; record the outcome. The run falls back to the generator automatically."""
    out = {}
    for name, url in PUBLIC_PROBES.items():
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URLs)
                out[name] = f"HTTP {r.status}"
        except Exception as e:  # noqa: BLE001
            out[name] = f"{type(e).__name__}: {e}"[:160]
    return out


def acquire(scale: Scale, probe: bool, log: Callable[[str], None]) -> dict:
    probe_result = probe_public() if probe else {"skipped": "sample scale uses the committed sample"}
    paths = generate.file_paths(scale)
    if scale.name == "sample" and paths["raw"].exists():
        manifest = json.loads(paths["manifest"].read_text())
    else:
        log("generating SYNTHETIC extract ...")
        manifest = generate.write(scale, public_probe=probe_result)
    manifest["public_probe"] = probe_result
    return manifest


def run(scale: Scale, results: Path, probe: bool = True, log: Callable[[str], None] = print) -> dict:
    t_all = time.time()
    timings: dict[str, float] = {}

    def tick(name: str, t0: float) -> float:
        timings[name] = round(time.time() - t0, 1)
        log(f"  {name}: {timings[name]} s")
        return time.time()

    results.mkdir(parents=True, exist_ok=True)
    t = time.time()
    manifest = acquire(scale, probe, log)
    paths = generate.file_paths(scale)
    t = tick("acquire", t)

    work = paths["raw"].parent if scale.name == "full" else results.parent / "sample_work"
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "civic311.duckdb"
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    n_raw = ingest.load_raw(con, paths["raw"])
    df = ingest.staged_frame(con)
    truth = pd.read_csv(paths["truth"], keep_default_na=False, dtype={"defect": str, "incident": str})
    truth["true_category"] = truth.true_category.replace("", None)
    assert len(truth) == len(df) == n_raw, "truth and raw must align row for row"
    t = tick("ingest_stage_duckdb", t)

    # ---------------------------------------------------------------- harmonise categories
    reviewed = categories.load_reviewed(REVIEWED_MAPPING)
    mapping = categories.build_mapping(df.sr_type_raw, reviewed)
    mapping_auto = categories.build_mapping(df.sr_type_raw, None)
    freq = df.sr_type_raw.value_counts()
    mapping["rows"] = mapping.raw_value.map(freq).fillna(0).astype(int)
    df["category"] = df.sr_type_raw.map(dict(zip(mapping.raw_value, mapping.category, strict=True)))
    df["category"] = df["category"].astype(object).where(df["category"].notna(), None)
    auto_only = df.sr_type_raw.map(dict(zip(mapping_auto.raw_value, mapping_auto.category, strict=True)))
    harmon = evaluate.harmonisation_accuracy(df, truth, auto_only)
    t = tick("harmonise", t)

    # ---------------------------------------------------------------- validate + quarantine
    rejects = contract.run_pandera(df)
    masks = contract.rule_masks(df)
    rederived = {(int(sl), code) for code, m in masks.items() for sl in df.source_line[m]}
    recorded = set(zip(rejects.source_line.astype(int), rejects.reason_code, strict=True))
    trace = {
        "quarantine_reason_rows": len(rejects),
        "reasons_with_valid_rule_id": int(rejects.rule_id.isin(contract.BY_ID).sum()),
        "reasons_rederived_by_independent_check": len(recorded & rederived),
        "independent_violations_missing_from_quarantine": len(rederived - recorded),
        "source_lines_found_in_raw": int(rejects.source_line.isin(df.source_line).sum()),
    }
    bad = set(rejects.source_line)
    clean = df[~df.source_line.isin(bad)].reset_index(drop=True)
    t = tick("pandera_validate_quarantine", t)

    ge_raw = contract.run_ge(df, "raw_extract")
    ge_clean = contract.run_ge(clean, "clean_table")
    t = tick("great_expectations", t)

    # persist the warehouse tables in DuckDB
    def join(s: pd.Series) -> str:
        return "|".join(sorted(s))

    qrows = (
        rejects.groupby("source_line")
        .agg(sr_number=("sr_number", "first"), n_reasons=("reason_code", "size"),
             reason_codes=("reason_code", join), rule_ids=("rule_id", join))
        .reset_index()
    )  # fmt: skip
    con.register("rejects_df", rejects)
    con.register("qrows_df", qrows)
    con.register("mapping_df", mapping)
    con.execute("CREATE OR REPLACE TABLE quarantine_reasons AS SELECT * FROM rejects_df")
    con.execute("CREATE OR REPLACE TABLE quarantine_rows AS SELECT * FROM qrows_df")
    con.execute("CREATE OR REPLACE TABLE category_mapping AS SELECT * FROM mapping_df")
    con.execute(
        "CREATE OR REPLACE TABLE clean AS SELECT s.* EXCLUDE (sr_type_raw), m.category, m.source AS category_source "
        "FROM staged s JOIN category_mapping m ON s.sr_type_raw = m.raw_value "
        "WHERE s.source_line NOT IN (SELECT source_line FROM quarantine_rows)"
    )
    reason_summary = con.execute(
        """
        SELECT q.rule_id, q.reason_code, count(*) AS reason_rows,
               count(*) FILTER (WHERE r.n_reasons = 1) AS sole_reason_rows,
               round(100.0 * count(*) / (SELECT count(*) FROM staged), 3) AS pct_of_extract
        FROM quarantine_reasons q JOIN quarantine_rows r USING (source_line)
        GROUP BY ALL ORDER BY reason_rows DESC
        """
    ).df()
    n_clean_db = con.execute("SELECT count(*) FROM clean").fetchone()[0]
    assert n_clean_db == len(clean), (n_clean_db, len(clean))
    examples = (
        rejects.groupby("reason_code", group_keys=False)
        .head(3)
        .merge(
            df[
                [
                    "source_line",
                    "sr_type_raw",
                    "status",
                    "created_raw",
                    "closed_raw",
                    "ward_raw",
                    "latitude",
                    "longitude",
                ]
            ],
            on="source_line",
        )
    )
    detection, det_summary = evaluate.defect_detection(truth, rejects)
    t = tick("persist_evaluate", t)

    # ---------------------------------------------------------------- KPIs and anomalies
    daily = kpi.daily_quality(df, rejects)
    q_anoms = kpi.quality_anomalies(daily)
    inc_table, inc_summary = kpi.score_incidents(q_anoms, generate.INCIDENTS)
    vol = kpi.volume_anomalies(df)
    storm = pd.to_datetime(list(generate.STORM_DAYS))
    vol_all = vol[vol.series == "ALL"]
    vol_summary = {
        "storm_days": [str(d.date()) for d in storm],
        "storm_days_flagged_total_volume": int(vol_all.date.isin(storm).sum()),
        "other_days_flagged_total_volume": int((~vol_all.date.isin(storm)).sum()),
        "category_day_flags": int(len(vol[vol.series != "ALL"])),
        "category_day_flags_on_storm_days": int(vol[(vol.series != "ALL") & vol.date.isin(storm)].shape[0]),
        "storm_categories_flagged": sorted(set(vol[(vol.series != "ALL") & vol.date.isin(storm)].series)),
        "max_total_volume_ratio": round(float(vol_all.ratio.max()), 2) if len(vol_all) else None,
    }
    slo = kpi.slo_burndown(daily)
    undated = int(len(df) - daily.rows.sum())
    t = tick("kpis_anomalies", t)

    # ---------------------------------------------------------------- hotspots
    res = hotspots.resolution_frame(clean)
    res["true_multiplier"] = truth.true_multiplier.to_numpy()[res.source_line.to_numpy() - 1]
    cells = hotspots.hotspot_table(res)
    cells_scored, hs_eval = hotspots.score_against_truth(cells, res)
    t = tick("h3_hotspots", t)

    # ---------------------------------------------------------------- equity (stretch)
    res["neighbourhood"] = equity.assign_neighbourhood(res)
    true_hood = truth.neighbourhood.to_numpy()[res.source_line.to_numpy() - 1]
    sjoin_agree = float((res.neighbourhood.to_numpy() == true_hood).mean())
    hot = set(cells.h3[cells.gi_class == "hot"])
    eq_table, eq_summary = equity.equity_analysis(res.dropna(subset=["neighbourhood"]), hot)
    eq_summary["sjoin_matches_generator_assignment"] = round(sjoin_agree, 5)
    eq_summary["injected_low_income_multiplier"] = generate.LOW_INCOME_MULT
    t = tick("equity", t)

    # ---------------------------------------------------------------- write results
    n = len(df)
    quarantined = len(bad)
    q_by_q = slo.groupby("quarter").last()
    metrics = {
        "data": {
            "path_used": manifest["path_used"], "label": "SYNTHETIC", "scale": scale.name, "seed": scale.seed,
            "public_probe": manifest["public_probe"], "rows_raw": n, "base_requests": manifest["base_requests"],
            "duplicates_appended": manifest["duplicates_appended"], "raw_file_bytes": manifest["raw_bytes"],
            "window": manifest["window"],
        },
        "contract": {
            "version": contract.CONTRACT_VERSION, "rules": len(contract.RULES),
            "ge_expectations": ge_raw["evaluated_expectations"],
        },
        "quarantine": {
            "rows_quarantined": quarantined, "pct_quarantined": round(100 * quarantined / n, 3),
            "rows_clean": len(clean), "reason_rows": len(rejects),
            "rows_with_multiple_reasons": int((qrows.n_reasons > 1).sum()),
        },
        "traceability": trace,
        "defect_detection": det_summary,
        "great_expectations": {
            "raw_success": ge_raw["success"], "raw_failed_expectations": ge_raw["evaluated_expectations"]
            - ge_raw["successful_expectations"], "clean_success": ge_clean["success"],
            "clean_failed_expectations": ge_clean["evaluated_expectations"] - ge_clean["successful_expectations"],
            "pandera_ge_agreement": report.agreement(rejects, ge_raw)["summary"],
        },
        "harmonisation": {
            "distinct_raw_values": len(mapping),
            "by_method": {k: int(v) for k, v in mapping.method.value_counts().items()},
            "reviewed_entries": len(reviewed),
            "accuracy": harmon,
        },
        "kpi_anomalies": {"quality": inc_summary, "volume": vol_summary, "undated_rows": undated,
                          "z_threshold": kpi.ANOMALY_Z, "min_lift": kpi.ANOMALY_MIN_LIFT},  # fmt: skip
        "slo": {
            "target_pass_rate": SLO_PASS_TARGET,
            "by_quarter": {
                q: {"pass_rate": round(1 - r.failed_cum / (r.budget_rows / (1 - SLO_PASS_TARGET)), 4),
                    "budget_left_pct": round(100 * r.budget_left, 1), "breached": bool(r.budget_left < 0)}
                for q, r in q_by_q.iterrows()
            },
        },  # fmt: skip
        "hotspots": {
            "h3_resolution": H3_RES, "min_cell_n": MIN_CELL_N, "fdr_q": FDR_Q, "material_excess": MATERIAL_EXCESS,
            "completed_requests_scored": int(len(res)), "cells_total": int(res.h3.nunique()),
            "cells_scored": int(len(cells)),
            "gi_classes": {k: int(v) for k, v in cells.gi_class.value_counts().items()},
            "naive_classes": {k: int(v) for k, v in cells.naive_class.value_counts().items()},
            "evaluation": hs_eval,
            "median_hours_hot": round(float(cells.loc[cells.gi_class == "hot", "median_hours"].median()), 1)
            if (cells.gi_class == "hot").any() else None,
            "median_hours_not_hot": round(float(cells.loc[cells.gi_class != "hot", "median_hours"].median()), 1),
            "requests_in_hot_cells": int(cells.loc[cells.gi_class == "hot", "n"].sum()),
        },  # fmt: skip
        "equity": eq_summary,
        "timings_s": timings,
    }  # fmt: skip

    report.write_all(
        results, metrics=metrics, rejects=rejects, reason_summary=reason_summary, examples=examples,
        detection=detection, mapping=mapping, reviewed=reviewed, ge_raw=ge_raw, ge_clean=ge_clean, daily=daily,
        q_anoms=q_anoms, inc_table=inc_table, vol=vol, slo=slo, cells=cells_scored, eq_table=eq_table, df=df,
    )  # fmt: skip
    con.close()
    metrics["timings_s"]["total"] = round(time.time() - t_all, 1)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n")
    return metrics


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))
