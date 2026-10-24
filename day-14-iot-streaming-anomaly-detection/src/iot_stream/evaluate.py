"""Warning evaluation (precision / recall / lead time) and streaming-vs-batch checks, in DuckDB SQL."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import HOUR, ModelConfig

RULES = ("combined", "hst_only", "rul_only")


def derive_warnings(rows: pd.DataFrame, cfg: ModelConfig, rule: str) -> pd.DataFrame:
    """Re-derive debounced warnings from logged per-window signals for a given rule.

    ``combined`` reproduces the online rule exactly (checked in tests).
    """
    out = []
    deb = cfg.debounce_h * HOUR
    for (m, _life), g in rows.sort_values("window_end").groupby(["machine", "life_start"], sort=False):
        last = -np.inf
        for t, hr, rr in zip(g["window_end"].to_numpy(), g["hst_run"].to_numpy(), g["rul_run"].to_numpy(), strict=True):
            h, r = hr >= cfg.hst_persist, rr >= cfg.rul_persist
            hit = (h or r) if rule == "combined" else (h if rule == "hst_only" else r)
            if hit and t - last >= deb:
                out.append((int(m), float(t)))
                last = t
    return pd.DataFrame(out, columns=["machine", "t"])


def evaluate_warnings(con: duckdb.DuckDBPyConnection, cfg: ModelConfig, horizon_s: float, n_machines: int) -> pd.DataFrame:
    """Expects tables ``warnings(variant, rule, machine, t)`` and ``failures``. Returns one row per variant x rule."""
    h = cfg.eval_horizon_h * HOUR
    warm = cfg.warmup_h * HOUR
    q = f"""
    WITH w AS (
        SELECT w.variant, w.rule, w.machine, w.t,
               EXISTS (SELECT 1 FROM failures f WHERE f.machine = w.machine
                       AND f.failure_time > w.t AND f.failure_time - w.t <= {h}) AS tp
        FROM warnings w
    ), prec AS (
        SELECT variant, rule, COUNT(*) AS n_warnings, SUM(tp::INT) AS tp_warnings FROM w GROUP BY ALL
    ), elig AS (
        SELECT * FROM failures WHERE failure_time >= {warm}
    ), hits AS (
        SELECT v.variant, v.rule, f.machine, f.failure_time, f.sudden,
               (SELECT MIN(w.t) FROM warnings w WHERE w.variant = v.variant AND w.rule = v.rule
                AND w.machine = f.machine AND w.t < f.failure_time AND w.t >= f.lifecycle_start
                AND f.failure_time - w.t <= {h}) AS first_warning
        FROM elig f CROSS JOIN (SELECT DISTINCT variant, rule FROM warnings) v
    ), rec AS (
        SELECT variant, rule, COUNT(*) AS n_failures,
               SUM((first_warning IS NOT NULL)::INT) AS detected,
               SUM((NOT sudden)::INT) AS n_gradual,
               SUM(((first_warning IS NOT NULL) AND NOT sudden)::INT) AS detected_gradual,
               SUM(sudden::INT) AS n_sudden,
               SUM(((first_warning IS NOT NULL) AND sudden)::INT) AS detected_sudden,
               MEDIAN((failure_time - first_warning) / 3600.0) AS median_lead_h,
               QUANTILE_CONT((failure_time - first_warning) / 3600.0, 0.25) AS p25_lead_h,
               QUANTILE_CONT((failure_time - first_warning) / 3600.0, 0.75) AS p75_lead_h,
               MIN((failure_time - first_warning) / 3600.0) AS min_lead_h
        FROM hits GROUP BY ALL
    )
    SELECT rec.variant, rec.rule, n_warnings, tp_warnings,
           tp_warnings::DOUBLE / NULLIF(n_warnings, 0) AS precision,
           detected::DOUBLE / NULLIF(n_failures, 0) AS recall,
           detected_gradual::DOUBLE / NULLIF(n_gradual, 0) AS recall_gradual,
           detected_sudden::DOUBLE / NULLIF(n_sudden, 0) AS recall_sudden,
           n_failures, detected, median_lead_h, p25_lead_h, p75_lead_h, min_lead_h,
           (n_warnings - tp_warnings)::DOUBLE / ({n_machines} * {horizon_s / 86400.0}) AS false_warnings_per_machine_day
    FROM rec JOIN prec USING (variant, rule)
    ORDER BY rec.variant, rec.rule
    """
    df = con.execute(q).df()
    p, r = df["precision"], df["recall"]
    df["f1"] = 2 * p * r / (p + r)
    return df


def lead_times(con: duckdb.DuckDBPyConnection, cfg: ModelConfig, variant: str, rule: str) -> pd.DataFrame:
    h = cfg.eval_horizon_h * HOUR
    return con.execute(
        f"""
        SELECT f.machine, f.failure_time, f.onset_time, f.sudden,
               (f.failure_time - MIN(w.t)) / 3600.0 AS lead_h
        FROM failures f JOIN warnings w ON w.machine = f.machine AND w.variant = ? AND w.rule = ?
             AND w.t < f.failure_time AND w.t >= f.lifecycle_start AND f.failure_time - w.t <= {h}
        WHERE f.failure_time >= {cfg.warmup_h * HOUR}
        GROUP BY f.machine, f.failure_time, f.onset_time, f.sudden ORDER BY f.failure_time
        """,
        [variant, rule],
    ).df()


def batch_window_check(con: duckdb.DuckDBPyConnection, slide_s: float) -> dict:
    """Recompute tumbling aggregates in batch SQL from accepted raw events and diff against the stream output.

    Expects tables ``events_accepted(machine, sensor, event_time, value)`` and ``stream_windows``.
    """
    res = con.execute(
        f"""
        WITH b AS (
            SELECT machine, (FLOOR(event_time / {slide_s}) + 1) * {slide_s} AS window_end,
                   COUNT(*) FILTER (WHERE sensor = 0) AS vib_count, AVG(value) FILTER (WHERE sensor = 0) AS vib_mean,
                   COUNT(*) FILTER (WHERE sensor = 1) AS temp_count, AVG(value) FILTER (WHERE sensor = 1) AS temp_mean,
                   COUNT(*) FILTER (WHERE sensor = 2) AS cur_count, AVG(value) FILTER (WHERE sensor = 2) AS cur_mean,
                   MAX(value) FILTER (WHERE sensor = 0) AS vib_max
            FROM events_accepted GROUP BY ALL
        )
        SELECT COUNT(*) AS batch_windows,
               (SELECT COUNT(*) FROM stream_windows) AS stream_windows,
               COUNT(s.machine) AS matched,
               SUM(CASE WHEN s.machine IS NULL OR b.vib_count <> s.vib_count OR b.temp_count <> s.temp_count
                        OR b.cur_count <> s.cur_count THEN 1 ELSE 0 END) AS count_mismatches,
               MAX(GREATEST(COALESCE(ABS(b.vib_mean - s.vib_mean), 0), COALESCE(ABS(b.temp_mean - s.temp_mean), 0),
                            COALESCE(ABS(b.cur_mean - s.cur_mean), 0), COALESCE(ABS(b.vib_max - s.vib_max), 0))) AS max_abs_diff
        FROM b LEFT JOIN stream_windows s USING (machine, window_end)
        """
    ).fetchone()
    keys = ["batch_windows", "stream_windows", "matched", "count_mismatches", "max_abs_diff"]
    return {k: (float(v) if k == "max_abs_diff" else int(v)) for k, v in zip(keys, res, strict=True)}


def rul_accuracy(con: duckdb.DuckDBPyConnection, cfg: ModelConfig, horizon_s: float) -> pd.DataFrame:
    """MAE of online RUL predictions against simulator ground truth (capped at ``rul_cap_h``), per variant.

    Expects ``scores(variant, machine, window_end, rul_pred, ...)``. Rows whose true RUL is right-censored
    below the cap (machine still running at the end of the horizon) are excluded.
    """
    cap = cfg.rul_cap_h
    return con.execute(
        f"""
        WITH s AS (
            SELECT s.variant, s.window_end, s.rul_pred, (f.failure_time - s.window_end) / 3600.0 AS rul_true_raw
            FROM scores s ASOF LEFT JOIN failures f ON s.machine = f.machine AND f.failure_time >= s.window_end
            WHERE s.rul_pred IS NOT NULL AND NOT isnan(s.rul_pred)
        ), t AS (
            SELECT variant, rul_pred, rul_true_raw,
                   CASE WHEN rul_true_raw IS NOT NULL THEN LEAST({cap}, rul_true_raw)
                        WHEN ({horizon_s} - window_end) / 3600.0 >= {cap} THEN {cap} END AS rul_true
            FROM s
        )
        SELECT variant,
               COUNT(rul_true) AS n_scored,
               AVG(ABS(rul_pred - rul_true)) AS mae_capped_h,
               COUNT(*) FILTER (WHERE rul_true_raw <= {cap}) AS n_within_cap,
               AVG(ABS(rul_pred - rul_true)) FILTER (WHERE rul_true_raw <= {cap}) AS mae_within_cap_h,
               AVG(ABS(rul_pred - rul_true)) FILTER (WHERE rul_true_raw <= 24) AS mae_last_24h_h
        FROM t GROUP BY ALL ORDER BY variant
        """
    ).df()
