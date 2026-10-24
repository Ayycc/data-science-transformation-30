"""End-to-end run: simulate -> stream (asyncio) -> online models -> DuckDB metrics -> results/."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import plots
from .config import HOUR, ModelConfig, SimConfig, StreamConfig
from .data import load_sample, probe_cmapss
from .evaluate import RULES, batch_window_check, derive_warnings, evaluate_warnings, lead_times, rul_accuracy
from .monitor import FleetMonitor
from .simulate import SensorStream, simulate
from .sink import exactly_once_demo
from .stream import expected_assignment, replay_models, run_end_to_end, run_windowing

ROOT = Path(__file__).resolve().parents[2]
SCORE_COLS = ["machine", "window_end", "life_start", "hst_score", "rul_pred", "hst_run", "rul_run", "warn"]
QUICK_MODEL = ModelConfig(
    hst_window=100, rul_min_labels=200, rul_min_cap_labels=200, ph_min_instances=200, static_freeze_day=1.5, warmup_h=6.0
)


def _lateness_stats(stream: SensorStream, cfg: StreamConfig, dead: list[int]) -> tuple[dict, np.ndarray]:
    lateness, expected = expected_assignment(stream, cfg)
    got = np.ones(len(stream), dtype=bool)
    got[np.asarray(dead, dtype=np.int64)] = False
    within = (lateness > 0) & (lateness <= cfg.allowed_lateness_s)
    beyond = lateness > cfg.allowed_lateness_s
    stats = {
        "events": int(len(stream)),
        "late_events_within_allowed_lateness": int(within.sum()),
        "late_within_lateness_assigned": int((within & got).sum()),
        "late_within_lateness_assigned_share": float((within & got).sum() / max(within.sum(), 1)),
        "events_beyond_allowed_lateness": int(beyond.sum()),
        "beyond_lateness_dropped": int((beyond & ~got).sum()),
        "beyond_lateness_still_assigned_window_open": int((beyond & got).sum()),
        "dropped_share": float((~got).mean()),
        "lateness_p50_s_of_late_events": float(np.median(lateness[lateness > 0])) if (lateness > 0).any() else 0.0,
        "lateness_p99_s_of_late_events": float(np.quantile(lateness[lateness > 0], 0.99)) if (lateness > 0).any() else 0.0,
        "lateness_over_60s_share": float((lateness > 60).mean()),
        "oracle_mismatches": int((expected != got).sum()),
    }
    return stats, lateness


def _scores_frame(mon: FleetMonitor) -> pd.DataFrame:
    df = pd.DataFrame(mon.rows, columns=SCORE_COLS)
    df["variant"] = mon.variant
    return df


def run(quick: bool = False, results_dir: Path | None = None, db_path: Path | None = None, make_plots: bool = True) -> dict:
    t_start = time.perf_counter()
    results_dir = Path(results_dir or ROOT / "results")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path or ROOT / "data" / "stream_metrics.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (db_path, Path(str(db_path) + ".wal")):
        if p.exists():
            p.unlink()

    scfg = StreamConfig()
    mcfg = QUICK_MODEL if quick else ModelConfig()

    # ------------------------------------------------------------------ data
    if quick:
        probe = {"reachable": None, "url": None, "note": "quick mode: committed SYNTHETIC sample, no network probe"}
        stream = load_sample()
        sim_cfg = None
    else:
        probe = probe_cmapss() if os.environ.get("DAY14_OFFLINE") != "1" else {"reachable": False, "url": None, "note": "DAY14_OFFLINE=1"}
        sim_cfg = SimConfig()
        stream = simulate(sim_cfg)
    source = {
        "primary_stream": "synthetic",
        "generator": stream.meta.get("generator"),
        "cmapss_probe": probe,
        "cmapss_replay_used": False,
        "note": "The IoT stream is SYNTHETIC by design (500 machines, out-of-order/late events, dropouts, drift). "
        + (
            "C-MAPSS was reachable but is only an optional replay (load_cmapss_as_stream); not used for reported numbers."
            if probe.get("reachable")
            else "C-MAPSS probe failed; seeded SYNTHETIC generator used (automatic fallback)."
        ),
    }
    print(f"[data] {len(stream):,} events, {len(stream.failures)} failures; C-MAPSS probe: {probe['note'][:120]}")

    # ------------------------------------------------------------------ stream processor only
    wrun = run_windowing(stream, scfg)
    records = wrun.window_records
    print(f"[window] {wrun.events_per_sec:,.0f} events/s, {len(records):,} window records")
    late_stats, lateness = _lateness_stats(stream, scfg, wrun.windower.dead_letter)

    con = duckdb.connect(str(db_path))
    con.register("fails_df", stream.failures)
    con.execute("CREATE TABLE failures AS SELECT * FROM fails_df")
    acc_mask = np.ones(len(stream), dtype=bool)
    acc_mask[np.asarray(wrun.windower.dead_letter, dtype=np.int64)] = False
    ev_df = pd.DataFrame(
        {
            "machine": stream.machine[acc_mask],
            "sensor": stream.sensor[acc_mask],
            "event_time": stream.event_time[acc_mask],
            "value": stream.value[acc_mask].astype(np.float64),
        }
    )
    con.execute("CREATE TABLE events_accepted AS SELECT * FROM ev_df")
    dead_df = pd.DataFrame({"offset": np.asarray(wrun.windower.dead_letter, dtype=np.int64)})
    dead_df["machine"] = stream.machine[dead_df["offset"]]
    dead_df["event_time"] = stream.event_time[dead_df["offset"]]
    dead_df["arrival"] = stream.arrival[dead_df["offset"]]
    con.execute("CREATE TABLE dead_letter AS SELECT * FROM dead_df")
    win_df = pd.DataFrame(records)
    con.execute("CREATE TABLE stream_windows AS SELECT * FROM win_df")
    check = batch_window_check(con, scfg.slide_s)
    con.execute("DROP TABLE events_accepted")
    del ev_df
    print(f"[check] stream vs batch SQL: {check}")

    # ------------------------------------------------------------------ end-to-end (adaptive) + variants
    monitors = {"adaptive": FleetMonitor(mcfg, "adaptive", scfg.slide_s)}
    e2e = run_end_to_end(stream, scfg, monitors["adaptive"])
    print(f"[e2e] {e2e.events_per_sec:,.0f} events/s end to end ({e2e.seconds:.1f} s)")
    model_secs = {}
    for v in ("no_reset", "static"):
        monitors[v] = FleetMonitor(mcfg, v, scfg.slide_s)
        model_secs[v] = replay_models(records, stream.failures, monitors[v])
        print(f"[model] {v}: {model_secs[v]:.1f} s")

    scores = pd.concat([_scores_frame(m) for m in monitors.values()], ignore_index=True)
    con.execute("CREATE TABLE scores AS SELECT * FROM scores")
    wl = []
    for v in monitors:
        sv = scores[scores.variant == v]
        for rule in RULES:
            d = derive_warnings(sv, mcfg, rule)
            d["variant"], d["rule"] = v, rule
            wl.append(d)
    warns = pd.concat(wl, ignore_index=True)[["variant", "rule", "machine", "t"]]
    con.execute("CREATE TABLE warnings AS SELECT * FROM warns")
    online = pd.DataFrame(monitors["adaptive"].warnings, columns=["machine", "t", "reason"])
    derived = warns[(warns.variant == "adaptive") & (warns.rule == "combined")]
    online_matches = bool(
        len(online) == len(derived) and set(zip(online.machine, online.t, strict=True)) == set(zip(derived.machine, derived.t, strict=True))
    )
    drift_df = pd.DataFrame(
        [(v, *e) for v, m in monitors.items() for e in m.drift_events], columns=["variant", "t", "detector", "channel", "action"]
    )
    con.execute("CREATE TABLE drift_events AS SELECT * FROM drift_df")
    med_df = pd.DataFrame(
        [(v, *e) for v, m in monitors.items() for e in m.fleet_medians], columns=["variant", "window_end", "vib_rel", "cur_rel", "temp_d"]
    )
    con.execute("CREATE TABLE fleet_medians AS SELECT * FROM med_df")

    ev = evaluate_warnings(con, mcfg, stream.horizon_s, stream.n_machines)
    rul = rul_accuracy(con, mcfg, stream.horizon_s)
    lead = lead_times(con, mcfg, "adaptive", "combined")
    n_records = len(records)
    tp = pd.DataFrame(
        [
            {
                "stage": "windowing only (source + event-time windows)",
                "events": len(stream),
                "seconds": wrun.seconds,
                "events_per_sec": wrun.events_per_sec,
            },
            {
                "stage": "end to end (windows + HST + RUL + drift)",
                "events": len(stream),
                "seconds": e2e.seconds,
                "events_per_sec": e2e.events_per_sec,
            },
        ]
    )
    con.execute("CREATE TABLE throughput AS SELECT * FROM tp")
    hourly = con.execute(
        """
        SELECT variant, FLOOR(t / 86400) AS day, COUNT(*) AS warnings FROM warnings WHERE rule = 'combined' GROUP BY ALL ORDER BY 1, 2
        """
    ).df()

    # ------------------------------------------------------------------ exactly-once demo (sample stream)
    sample = load_sample()
    eo = exactly_once_demo(sample, replace(scfg, batch_size=1024), checkpoint_every=5, crash_after_batch=23)
    print(f"[exactly-once] {eo}")

    # ------------------------------------------------------------------ outputs
    ev.round(4).to_csv(results_dir / "warning_eval.csv", index=False)
    rul.round(3).to_csv(results_dir / "rul_accuracy.csv", index=False)
    lead.round(3).to_csv(results_dir / "lead_times_adaptive.csv", index=False)
    drift_df.assign(day=(drift_df.t / 86400).round(3)).to_csv(results_dir / "drift_events.csv", index=False)
    tp.round(3).to_csv(results_dir / "throughput.csv", index=False)
    hourly.to_csv(results_dir / "warnings_per_day.csv", index=False)

    head = ev[(ev.variant == "adaptive") & (ev.rule == "combined")].iloc[0]
    metrics = {
        "data": {
            "source": source,
            "stream": stream.meta,
            "simulation_config": asdict(sim_cfg) if sim_cfg else "data/sample",
            "scale": f"{stream.n_machines} machines x {stream.horizon_s / 86400:.0f} days",
        },
        "stream_config": asdict(scfg),
        "model_config": {k: v for k, v in asdict(mcfg).items()},
        "windowing": {**wrun.windower.counters(), **late_stats, "stream_vs_batch_sql": check},
        "throughput": {
            "windowing_events_per_sec": wrun.events_per_sec,
            "end_to_end_events_per_sec": e2e.events_per_sec,
            "end_to_end_seconds": e2e.seconds,
            "window_records": n_records,
            "model_stage_records_per_sec": {v: n_records / s for v, s in model_secs.items()},
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "note": "single asyncio event loop on one core",
        },
        "headline": {
            "variant": "adaptive",
            "rule": "combined",
            "precision": float(head.precision),
            "recall": float(head.recall),
            "recall_gradual_failures": float(head.recall_gradual),
            "recall_sudden_failures": float(head.recall_sudden),
            "median_lead_time_h": float(head.median_lead_h),
            "p25_lead_time_h": float(head.p25_lead_h),
            "p75_lead_time_h": float(head.p75_lead_h),
            "false_warnings_per_machine_day": float(head.false_warnings_per_machine_day),
            "n_failures_evaluated": int(head.n_failures),
            "online_rule_matches_rederived": online_matches,
        },
        "warning_eval": ev.round(4).to_dict(orient="records"),
        "rul_accuracy": rul.round(3).to_dict(orient="records"),
        "monitors": {v: m.summary() for v, m in monitors.items()},
        "drift_detections_by_variant": drift_df.groupby(["variant", "detector"]).size().rename("n").reset_index().to_dict(orient="records"),
        "exactly_once_demo": eo,
        "duckdb": {
            "path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
            "tables": [r[0] for r in con.execute("SHOW TABLES").fetchall()],
        },
        "acceptance": {
            "late_events_within_lateness_correctly_assigned": bool(
                late_stats["oracle_mismatches"] == 0
                and late_stats["late_within_lateness_assigned_share"] == 1.0
                and check["count_mismatches"] == 0
            ),
            "median_lead_time_positive": bool(head.median_lead_h > 0),
            "throughput_at_least_10k_eps": bool(e2e.events_per_sec >= 10_000),
        },
        "runtime_seconds": None,
    }

    if make_plots:
        win_small = win_df[["machine", "window_end", "vib_mean", "temp_mean"]]
        sc = con.execute(
            f"""SELECT s.machine, s.window_end, s.hst_score, s.rul_pred, LEAST({mcfg.rul_cap_h}, (f.failure_time - s.window_end) / 3600.0) AS rul_true
                FROM scores s ASOF LEFT JOIN failures f ON s.machine = f.machine AND f.failure_time >= s.window_end
                WHERE s.variant = 'adaptive'"""
        ).df()
        wa = warns[(warns.variant == "adaptive") & (warns.rule == "combined")]
        ex = _example_machines(lead, stream.failures)
        if ex:
            plots.machine_trajectory(win_small, sc, wa, stream.failures, ex[0], mcfg.hst_threshold, fig_dir / "machine_trajectory.png")
            plots.dashboard(win_small, sc, wa, stream.failures, ex[:8], mcfg.hst_threshold, results_dir / "dashboard.html")
        if len(lead):
            plots.lead_time_hist(lead, mcfg.eval_horizon_h, fig_dir / "lead_time_hist.png")
        plots.drift_timeline(med_df, drift_df, stream.meta.get("regime_change_time_s", 0), fig_dir / "drift_timeline.png")
        plots.variant_bars(ev, fig_dir / "variant_comparison.png")
        plots.throughput_bars(tp, fig_dir / "throughput.png")
        plots.lateness_hist(lateness, scfg.allowed_lateness_s, fig_dir / "lateness_hist.png")
    con.close()
    metrics["runtime_seconds"] = time.perf_counter() - t_start
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    _report(metrics, ev, rul, tp, results_dir / "report.md")
    print(f"[done] {metrics['runtime_seconds']:.0f} s")
    return metrics


def _example_machines(lead: pd.DataFrame, failures: pd.DataFrame) -> list[int]:
    """Machines with a detected gradual failure (ordered by lead time closest to the median), for figures."""
    if len(lead) == 0:
        return sorted(failures.machine.unique().tolist())[:8]
    g = lead[~lead.sudden].copy()
    g["d"] = (g.lead_h - g.lead_h.median()).abs()
    return [int(m) for m in g.sort_values("d").machine.drop_duplicates().tolist()]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append(
            "| "
            + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:,.3f}" if isinstance(v, float) else str(v)) for v in row)
            + " |"
        )
    return "\n".join(out)


def _report(m: dict, ev: pd.DataFrame, rul: pd.DataFrame, tp: pd.DataFrame, path: Path) -> None:
    w = m["windowing"]
    lines = [
        "# Day 14 technical report (SYNTHETIC data)",
        "",
        f"Scale: {m['data']['scale']}, {m['data']['stream']['n_events']:,} raw sensor events, {m['data']['stream']['n_failures']} failures.",
        "",
        "## Warning evaluation (all variants x rules)",
        "",
        _md(ev.round(3)),
        "",
        "## Online RUL accuracy (hours, vs simulator ground truth capped at 72 h)",
        "",
        _md(rul.round(2)),
        "",
        "## Throughput",
        "",
        _md(tp.round(1)),
        "",
        "## Windowing correctness",
        "",
        "| check | value |",
        "|---|---|",
    ]
    for k in (
        "events_in",
        "events_out_of_order",
        "late_events_within_allowed_lateness",
        "late_within_lateness_assigned",
        "events_beyond_allowed_lateness",
        "beyond_lateness_dropped",
        "beyond_lateness_still_assigned_window_open",
        "oracle_mismatches",
        "window_records_fired",
    ):
        lines.append(f"| {k} | {w[k]:,} |")
    for k, v in w["stream_vs_batch_sql"].items():
        lines.append(f"| stream_vs_batch_sql.{k} | {v} |")
    lines += ["", "## Exactly-once sink demo (sample stream)", "", "```json", json.dumps(m["exactly_once_demo"], indent=2), "```", ""]
    path.write_text("\n".join(lines))


__all__ = ["run", "HOUR"]
