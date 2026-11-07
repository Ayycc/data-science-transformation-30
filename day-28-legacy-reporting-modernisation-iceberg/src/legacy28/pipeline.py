"""End-to-end run: generate SYNTHETIC extracts, run 8 quarterly closes in parallel
(legacy vs modern), reconcile, write Iceberg tables, verify time travel, benchmark,
and write results/."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import pyarrow as pa

from . import figures, playbook
from .config import IBNR_FROM_QUARTER, QUARTERS, SEED
from .generator import Extracts, as_of_close, generate
from .iceberg_store import REPORT_V1, Lakehouse, df_to_arrow
from .io import extracts_to_arrow, to_arrow
from .legacy import run_legacy
from .modern import TARGET, attribution_steps, run_modern, run_modern_arrow
from .reconcile import KEYS, affected_sets, check_step, compare_summary, datacompy_compare, diff_cells
from .rules import DIFFERENCES, RULES, inventory_markdown, lineage_markdown, topo_order

ROOT = Path(__file__).resolve().parents[2]


def _increment(cur: pd.DataFrame, prev: pd.DataFrame | None, key: list[str]) -> pd.DataFrame:
    if prev is None or prev.empty:
        return cur
    m = cur.merge(prev[key], on=key, how="left", indicator=True)
    return cur[(m["_merge"] == "left_only").to_numpy()]


RAW_KEYS = {"policy_versions": ["load_seq"], "premium_txns": ["txn_id"], "claim_valuations": ["claim_id", "valuation_q"]}


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> bool:
    return diff_cells(a, b, keys).empty and len(a) == len(b)


def write_sample(ex: Extracts, out: Path, n_pol: int = 1_500) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, df in ex.as_dict().items():
        sub = df[df["policy_id"] <= n_pol]
        p = out / f"{name}_SYNTHETIC_sample.csv"
        sub.to_csv(p, index=False)
        sizes[p.name] = p.stat().st_size
    return sizes


def parallel_run(ex: Extracts, lake: Lakehouse, log=print) -> dict:
    closes = []
    filed_modern, filed_legacy, full_reports = {}, {}, {}
    schema_events = []
    prev_asof = None
    trace_all = None
    for q in QUARTERS:
        label = q.label
        asof = as_of_close(ex, q.idx)
        # 1. land the newly booked source rows in the Iceberg raw layer
        for name, df in asof.as_dict().items():
            inc = _increment(df, None if prev_asof is None else prev_asof.as_dict()[name], RAW_KEYS[name])
            lake.append_raw(name, to_arrow(name, inc), label)
        prev_asof = asof
        sources = {name: lake.read_raw(name) for name in ("policy_versions", "premium_txns", "claim_valuations")}
        src_ok = all(sources[k].num_rows == len(v) for k, v in asof.as_dict().items())

        # 2. legacy run (reference)
        t0 = time.perf_counter()
        legacy = run_legacy(asof, q.idx)
        t_legacy = time.perf_counter() - t0

        # 3. modern runs: compat, then each intentional difference switched on in turn
        aff = affected_sets(asof, q.idx)
        con = ibis.duckdb.connect()
        steps, prev, compat_cmp = [], None, {}
        step_diffs = {}
        t_target = None
        for sid, cfg in attribution_steps():
            t0 = time.perf_counter()
            cur = run_modern(sources, q.idx, cfg, "duckdb", con)
            if sid == "D6":
                t_target = time.perf_counter() - t0
            if prev is None:
                compat_cmp = {t: compare_summary(datacompy_compare(legacy[t], cur[t], KEYS[t])) for t in KEYS}
            else:
                r = check_step(sid, prev, cur, aff)
                steps.append(r)
                if q.idx == len(QUARTERS):
                    step_diffs[sid] = {t: diff_cells(prev[t], cur[t], KEYS[t]) for t in KEYS}
            prev = cur
        target = prev
        polars = run_modern(sources, q.idx, TARGET, "polars")
        engines_equal = all(_frames_equal(target[t], polars[t], KEYS[t]) for t in KEYS)
        final_cmp = {t: compare_summary(datacompy_compare(legacy[t], target[t], KEYS[t])) for t in KEYS}
        unexplained = sum(v["unequal_cells"] + v["rows_only_legacy"] + v["rows_only_modern"] for v in compat_cmp.values())
        unexplained += sum(sum(r.unexplained.values()) for r in steps)

        # 4. write the modern outputs to Iceberg
        rep = target["report"]
        with_ibnr = q.idx >= IBNR_FROM_QUARTER
        rep_to_write = rep if with_ibnr else rep.drop(columns=["ibnr_cents"])
        schema_events += lake.write_report(rep_to_write, label, with_ibnr)
        lake.write_detail("premium_detail", target["premium_detail"], label)
        lake.write_detail("claims_detail", target["claims_detail"], label)
        full_reports[label] = rep_to_write
        filed_modern[label] = rep[rep["quarter"] == label].reset_index(drop=True)
        filed_legacy[label] = legacy["report"][legacy["report"]["quarter"] == label].reset_index(drop=True)

        if q.idx == len(QUARTERS):
            # trace every final legacy-vs-target cell difference to the step that introduced it
            trace_all = {}
            for t in KEYS:
                fd = diff_cells(legacy[t], target[t], KEYS[t])
                fkeys = set(map(tuple, fd[[*KEYS[t], "column"]].astype(str).to_numpy()))
                seen = set()
                for sid in step_diffs:
                    sd = step_diffs[sid][t]
                    seen |= set(map(tuple, sd[[*KEYS[t], "column"]].astype(str).to_numpy()))
                trace_all[t] = {"final_diff_cells": len(fkeys), "traced_to_a_step": len(fkeys & seen), "untraced": len(fkeys - seen)}
            final_legacy, final_target = legacy, target

        lt, mt = legacy["report"], target["report"]
        row = {
            "close": label,
            "source_rows_match_iceberg": bool(src_ok),
            "legacy_seconds": round(t_legacy, 3),
            "modern_duckdb_seconds": round(t_target, 3),
            "compat_equals_legacy": all(v["matches"] for v in compat_cmp.values()),
            "duckdb_equals_polars": bool(engines_equal),
            "unexplained_differences": int(unexplained),
            "report_rows": int(len(mt)),
            "report_cells_changed_by": {r.step: r.changed_cells["report"] for r in steps},
            "detail_rows_changed_by": {r.step: r.changed_rows["premium_detail"] + r.changed_rows["claims_detail"] for r in steps},
            "final_vs_legacy": final_cmp,
            "filed_quarter_totals": {
                "legacy_earned_cents": int(lt.loc[lt.quarter == label, "earned_premium_cents"].sum()),
                "modern_earned_cents": int(mt.loc[mt.quarter == label, "earned_premium_cents"].sum()),
                "legacy_incurred_cents": int(lt.loc[lt.quarter == label, "incurred_cents"].sum()),
                "modern_incurred_cents": int(mt.loc[mt.quarter == label, "incurred_cents"].sum()),
            },
        }
        closes.append(row)
        log(f"[close {label}] legacy {t_legacy:.2f}s modern {t_target:.2f}s compat==legacy {row['compat_equals_legacy']} "
            f"duckdb==polars {engines_equal} unexplained {unexplained}")  # fmt: skip
    return {
        "closes": closes, "filed_modern": filed_modern, "filed_legacy": filed_legacy, "full_reports": full_reports,
        "schema_events": schema_events, "trace": trace_all, "final_legacy": final_legacy, "final_target": final_target,
    }  # fmt: skip


def time_travel_checks(lake: Lakehouse, pr: dict, log=print) -> dict:
    tag_ok, filed_ok, input_ok, schema_ok = 0, 0, 0, 0
    restated = {}
    current = lake.read_report()
    for q in QUARTERS:
        label = q.label
        snap = lake.read_report(label)
        keys = ["quarter", "lob", "region"]
        snap = snap.astype({c: object for c in snap.columns if str(snap[c].dtype) not in ("Int64",)})
        tag_ok += _frames_equal(snap.sort_values(keys).reset_index(drop=True), pr["full_reports"][label], keys)
        filed = snap[snap["quarter"] == label].sort_values(keys).reset_index(drop=True)
        ref = pr["filed_modern"][label]
        ref = ref if "ibnr_cents" in filed.columns else ref.drop(columns=["ibnr_cents"])
        filed_ok += _frames_equal(filed, ref, keys)
        # re-run the modern pipeline from the raw tables as they were at that close
        sources = {n: lake.read_raw(n, label) for n in ("policy_versions", "premium_txns", "claim_valuations")}
        rerun = run_modern(sources, q.idx, TARGET, "duckdb")["report"]
        rerun = rerun if "ibnr_cents" in snap.columns else rerun.drop(columns=["ibnr_cents"])
        input_ok += _frames_equal(rerun, pr["full_reports"][label], keys)
        cols = lake.report_schema_at(label)
        schema_ok += ("ibnr_cents" in cols) == (q.idx >= IBNR_FROM_QUARTER)
        # how much did later bookings restate this quarter in today's table?
        cur_q = current[current["quarter"] == label].astype({"quarter": object, "lob": object, "region": object})
        cur_q = cur_q[filed.columns].sort_values(keys).reset_index(drop=True)
        d = diff_cells(filed, cur_q, keys)
        restated[label] = {
            "cells_restated": int(len(d)),
            "earned_restatement_cents": int(cur_q["earned_premium_cents"].sum() - filed["earned_premium_cents"].sum()),
            "written_restatement_cents": int(cur_q["written_premium_cents"].sum() - filed["written_premium_cents"].sum()),
        }
    # rollback drill: a bad run overwrites the report, Iceberg rolls back to the last good close
    tbl = lake.table("reporting.quarterly_report")
    good = tbl.current_snapshot().snapshot_id
    bad = pr["full_reports"][QUARTERS[-1].label].copy()
    bad["earned_premium_cents"] = bad["earned_premium_cents"] * 0

    tbl.overwrite(df_to_arrow(bad, REPORT_V1.append(pa.field("ibnr_cents", pa.int64()))), snapshot_properties={"close": "BAD-RUN"})
    bad_seen = int(lake.read_report()["earned_premium_cents"].sum()) == 0
    lake.table("reporting.quarterly_report").manage_snapshots().rollback_to_snapshot(good).commit()
    after = lake.read_report().astype({"quarter": object, "lob": object, "region": object}).sort_values(["quarter", "lob", "region"]).reset_index(drop=True)
    rollback_ok = bad_seen and _frames_equal(after, pr["full_reports"][QUARTERS[-1].label], ["quarter", "lob", "region"])
    tbl = lake.table("reporting.quarterly_report")
    res = {
        "closes": len(QUARTERS),
        "tag_snapshot_equals_report_written": int(tag_ok),
        "filed_quarter_reproduced": int(filed_ok),
        "rerun_from_time_travelled_inputs_reproduces_report": int(input_ok),
        "schema_version_correct_at_each_close": int(schema_ok),
        "previous_quarter": QUARTERS[-2].label,
        "previous_quarter_reproduced": bool(filed_ok == len(QUARTERS)),
        "restatements_in_current_table": restated,
        "rollback_drill_ok": bool(rollback_ok),
        "report_snapshots": len(tbl.snapshots()),
        "report_schemas": len(tbl.schemas()),
        "schema_events": pr["schema_events"],
    }
    log(f"[time travel] filed reproduced {filed_ok}/8, from inputs {input_ok}/8, rollback {rollback_ok}")
    return res


def benchmark(scales: list[int], base_n: int, reps: int, log=print) -> list[dict]:
    rows = []
    for s in scales:
        ex = as_of_close(generate(base_n * s, seed=SEED + s), len(QUARTERS))
        src = extracts_to_arrow(ex)
        n_detail = None
        for engine in ("legacy", "ibis-duckdb", "ibis-polars"):
            walls, cpus = [], []
            for _ in range(reps if engine != "legacy" or s == 1 else 1):
                w0, c0 = time.perf_counter(), time.process_time()
                if engine == "legacy":
                    out = run_legacy(ex, len(QUARTERS))
                else:
                    out = run_modern_arrow(src, len(QUARTERS), TARGET, engine.split("-")[1])
                walls.append(time.perf_counter() - w0)
                cpus.append(time.process_time() - c0)
            n_detail = sum(len(out[k]) if engine == "legacy" else out[k].num_rows for k in ("premium_detail", "claims_detail"))
            rows.append({"scale": s, "policies": base_n * s, "engine": engine, "wall_seconds": round(statistics.median(walls), 3),
                         "cpu_seconds": round(statistics.median(cpus), 3), "detail_rows": n_detail, "reps": len(walls)})  # fmt: skip
            log(f"[bench x{s}] {engine}: {rows[-1]['wall_seconds']}s wall, {rows[-1]['cpu_seconds']}s cpu")
    return rows


def storage_footprint(ex: Extracts, lake: Lakehouse, tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    csv_bytes = 0
    for name, df in as_of_close(ex, len(QUARTERS)).as_dict().items():
        p = tmp / f"{name}.csv"
        df.to_csv(p, index=False)
        csv_bytes += p.stat().st_size
        p.unlink()
    tmp.rmdir()
    return {"legacy_csv_extract_bytes": csv_bytes, "iceberg_raw_parquet_bytes": lake.data_bytes("raw.db") or lake.data_bytes("raw"),
            "iceberg_reporting_parquet_bytes_all_snapshots": lake.data_bytes("reporting.db") or lake.data_bytes("reporting")}  # fmt: skip


def main(n_policies: int = 30_000, scales: list[int] | None = None, reps: int = 3, results: Path | None = None, data: Path | None = None) -> dict:
    t_start = time.perf_counter()
    results = results or ROOT / "results"
    data = data or ROOT / "data"
    scales = scales or [1, 4, 10]
    (results / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[data] generating SYNTHETIC extracts: {n_policies:,} policies, seed {SEED}")
    ex = generate(n_policies, SEED)
    sample_sizes = write_sample(ex, data / "sample")
    lake = Lakehouse(data / "lakehouse", fresh=True)

    pr = parallel_run(ex, lake)
    tt = time_travel_checks(lake, pr)
    store = storage_footprint(ex, lake, data / "tmp_csv")
    bench = benchmark(scales, n_policies, reps)

    fl, ft = pr["final_legacy"]["report"], pr["final_target"]["report"]
    closes = pr["closes"]
    last = closes[-1]
    b1 = {r["engine"]: r for r in bench if r["scale"] == scales[0]}
    bmax = {r["engine"]: r for r in bench if r["scale"] == scales[-1]}
    impact = {
        "earned_premium_cents": {"legacy": int(fl.earned_premium_cents.sum()), "modern": int(ft.earned_premium_cents.sum())},
        "incurred_cents": {"legacy": int(fl.incurred_cents.sum()), "modern": int(ft.incurred_cents.sum())},
        "ibnr_cents": {"legacy": int(fl.ibnr_cents.dropna().sum()), "modern": int(ft.ibnr_cents.dropna().sum())},
        "other_lob_earned_cents": {"legacy": int(fl.loc[fl.lob == "OTHER", "earned_premium_cents"].sum()), "modern": int(ft.loc[ft.lob == "OTHER", "earned_premium_cents"].sum())},
    }  # fmt: skip
    step_impact = _step_totals(ex, pr)
    metrics = {
        "data": {
            "path_used": "synthetic_generator",
            "public_source_attempted": False,
            "note": "No public policy-level insurer dataset exists; the prompt specifies synthetic policy, premium and claims tables.",
            "seed": SEED,
            "policies": n_policies,
            "policy_versions": int(len(ex.policy_versions)),
            "premium_txns": int(len(ex.premium_txns)),
            "claim_valuations": int(len(ex.claim_valuations)),
            "claims": int(ex.claim_valuations.claim_id.nunique()),
            "sample_files_bytes": sample_sizes,
        },
        "rules": {"n_rules": len(RULES), "n_intentional_differences": len(DIFFERENCES), "topological_order": topo_order()},
        "parallel_run": {
            "closes": len(closes),
            "closes_compat_equals_legacy": sum(c["compat_equals_legacy"] for c in closes),
            "closes_duckdb_equals_polars": sum(c["duckdb_equals_polars"] for c in closes),
            "unexplained_differences_total": sum(c["unexplained_differences"] for c in closes),
            "final_close": last["close"],
            "final_close_vs_legacy": last["final_vs_legacy"],
            "final_close_report_cells_changed_by": last["report_cells_changed_by"],
            "final_close_detail_rows_changed_by": last["detail_rows_changed_by"],
            "final_close_trace": pr["trace"],
            "per_close": closes,
        },
        "impact_final_close_all_quarters": impact,
        "impact_by_difference_final_close": step_impact,
        "time_travel": tt,
        "storage": store,
        "benchmark": bench,
        "benchmark_summary": {
            "scale_1_speedup_duckdb_vs_legacy": round(b1["legacy"]["wall_seconds"] / b1["ibis-duckdb"]["wall_seconds"], 1),
            "scale_1_speedup_polars_vs_legacy": round(b1["legacy"]["wall_seconds"] / b1["ibis-polars"]["wall_seconds"], 1),
            "scale_max_speedup_duckdb_vs_legacy": round(bmax["legacy"]["wall_seconds"] / bmax["ibis-duckdb"]["wall_seconds"], 1),
            "scale_max_speedup_polars_vs_legacy": round(bmax["legacy"]["wall_seconds"] / bmax["ibis-polars"]["wall_seconds"], 1),
        },
        "environment": {"python": platform.python_version(), "ibis": ibis.__version__, "cpu_count": __import__("os").cpu_count()},
    }
    metrics["acceptance"] = {
        "reconciliation_zero_unexplained": metrics["parallel_run"]["unexplained_differences_total"] == 0
        and all(v["untraced"] == 0 for v in pr["trace"].values()),
        "same_ibis_code_two_backends_identical": metrics["parallel_run"]["closes_duckdb_equals_polars"] == len(closes),
        "time_travel_reproduces_previous_quarter": tt["previous_quarter_reproduced"]
        and tt["rerun_from_time_travelled_inputs_reproduces_report"] == len(QUARTERS),
    }
    metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 1)

    ft.to_csv(results / "quarterly_report_modern_2025Q4_close.csv", index=False)
    pd.DataFrame(bench).to_csv(results / "benchmark.csv", index=False)
    (results / "rule_inventory.md").write_text(inventory_markdown())
    (results / "lineage.md").write_text(lineage_markdown())
    cmp = datacompy_compare(fl, ft, KEYS["report"])
    (results / "reconciliation_datacompy_2025Q4.txt").write_text(
        "SYNTHETIC data. datacompy report: legacy vs modern TARGET quarterly report at the 2025Q4 close.\n"
        "Every difference below is attributed to D1-D6 in reconciliation.md.\n\n" + cmp.report(sample_count=5)
    )
    (results / "reconciliation.md").write_text(playbook.reconciliation_md(metrics))
    (results / "migration_playbook.md").write_text(playbook.playbook_md(metrics))
    figures.make_all(metrics, results / "figures")
    metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    print(f"[done] {metrics['runtime_seconds']} s; acceptance: {metrics['acceptance']}")
    return metrics


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _step_totals(ex: Extracts, pr: dict) -> dict:
    """Totals over all quarters at the final close after each cumulative step (from re-running the steps)."""
    asof = as_of_close(ex, len(QUARTERS))
    src = extracts_to_arrow(asof)
    out, prev = {}, None
    con = ibis.duckdb.connect()
    for sid, cfg in attribution_steps():
        rep = run_modern(src, len(QUARTERS), cfg, "duckdb", con)["report"]
        tot = {c: int(rep[c].dropna().sum()) for c in ("earned_premium_cents", "incurred_cents", "ibnr_cents")}
        tot["other_lob_earned_cents"] = int(rep.loc[rep.lob == "OTHER", "earned_premium_cents"].sum())
        if prev is not None:
            out[sid] = {k: tot[k] - prev[k] for k in tot}
        prev = tot
    return out
