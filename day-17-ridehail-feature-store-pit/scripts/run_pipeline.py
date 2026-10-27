"""End-to-end Day 17 run: data -> DuckDB features -> Feast (DuckDB offline, SQLite online)
-> naive vs point-in-time training sets -> LightGBM -> online replay, skew test, latency.

Writes everything quoted in the README to results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridehail_fs import data as D  # noqa: E402
from ridehail_fs import features as F  # noqa: E402
from ridehail_fs import joins as J  # noqa: E402
from ridehail_fs import model as M  # noqa: E402
from ridehail_fs import plots as P  # noqa: E402
from ridehail_fs import serving as S  # noqa: E402
from ridehail_fs.config import Paths, SimConfig  # noqa: E402
from ridehail_fs.feature_repo import make_store  # noqa: E402


def log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="shrink the simulation (smoke runs)")
    ap.add_argument("--offline", action="store_true", help="skip the NYC TLC download attempt")
    ap.add_argument("--checkpoints", type=int, default=80)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = SimConfig()
    if args.days:
        cfg = replace(cfg, n_days=args.days, train_days=max(2, int(args.days * 0.75)), warmup_days=1)
    paths = Paths()
    for p in (paths.offline, paths.store):
        shutil.rmtree(p, ignore_errors=True)
    paths.figures.mkdir(parents=True, exist_ok=True)

    # 1. data
    prov = D.acquire(cfg, paths, allow_network=not args.offline)
    tables = D.load_tables(paths)
    trips = tables["trips"]
    log(f"data: {prov['source']} {prov['rows']}", t0)

    # 2. features (DuckDB) -> parquet offline store
    tf = time.perf_counter()
    feats = F.build_feature_tables(tables, cfg.tick_min)
    F.write_feature_tables(feats, paths.offline)
    feat_build_s = time.perf_counter() - tf
    snaps = F.hourly_snapshot_tables(tables, feats)
    log(f"features: { {k: len(v) for k, v in feats.items()} }", t0)

    # 3. Feast repo
    store, defs = make_store(paths.store, paths.offline)
    log("feast apply done", t0)

    # 4. splits (labels must be known before the training cut-off)
    t_start = trips.request_ts.min().normalize()
    warm = t_start + pd.Timedelta(days=cfg.warmup_days)
    cut = t_start + pd.Timedelta(days=cfg.train_days)
    train_mask = (trips.request_ts >= warm) & (trips.dropoff_ts < cut)
    test_mask = trips.request_ts >= cut + pd.Timedelta(minutes=0)
    test_mask &= trips.request_ts < t_start + pd.Timedelta(days=cfg.n_days)
    use = trips[train_mask | test_mask].reset_index(drop=True)
    is_train = use.trip_id.isin(trips.trip_id[train_mask]).to_numpy()

    # 5. three joins
    tj = time.perf_counter()
    naive = J.naive_join(use, snaps)
    naive_s = time.perf_counter() - tj
    tj = time.perf_counter()
    pit = J.feast_pit_join(store, use)
    pit_s = time.perf_counter() - tj
    ref = J.reference_asof(use, feats)
    log(f"joins: naive {naive_s:.1f}s, feast PIT {pit_s:.1f}s", t0)
    pit_check = J.compare_frames(pit, ref, J.ALL_FEATURES)
    naive_vs_pit = J.compare_frames(naive, pit, J.ALL_FEATURES, rtol=1e-6)

    # leakage diagnostics: how far into the future each naive feature row reaches
    hour_end = use.request_ts.dt.floor("h") + pd.Timedelta(hours=1)
    zone_lookahead = (hour_end - use.request_ts).dt.total_seconds() / 60.0
    export_end = trips.dropoff_ts.max()
    driver_lookahead_h = (export_end - use.request_ts).dt.total_seconds() / 3600.0
    trip_in_window = (use.request_ts + pd.to_timedelta(use.duration_min, unit="m") <= hour_end).mean()

    # 6. models
    y = use.set_index("trip_id").duration_min
    naive = naive.set_index("trip_id").loc[use.trip_id].reset_index()
    pit = pit.set_index("trip_id").loc[use.trip_id].reset_index()
    ytr, yte = y.loc[use.trip_id[is_train]], y.loc[use.trip_id[~is_train]]
    m_naive = M.train(naive[is_train], ytr, cfg.seed)
    m_pit = M.train(pit[is_train], ytr, cfg.seed)
    log("models trained", t0)
    res = {
        "naive": {
            "offline": M.metrics(yte, M.predict(m_naive, naive[~is_train])),
            "production": M.metrics(yte, M.predict(m_naive, pit[~is_train])),
        },
        "pit": {
            "offline": M.metrics(yte, M.predict(m_pit, pit[~is_train])),
            "production": M.metrics(yte, M.predict(m_pit, pit[~is_train])),
        },
        "heuristic_driver_adjusted_eta": M.metrics(yte, pit[~is_train].driver_adjusted_eta_min.to_numpy()),
    }
    for k in ("naive", "pit"):
        off, prod = res[k]["offline"]["mae_min"], res[k]["production"]["mae_min"]
        res[k]["gap_mae_min"] = prod - off
        res[k]["gap_pct"] = (prod - off) / off * 100.0
    res["naive_vs_pit_production_mae_pct"] = (res["naive"]["production"]["mae_min"] / res["pit"]["production"]["mae_min"] - 1.0) * 100.0
    imp = {}
    for k, m in (("naive", m_naive), ("pit", m_pit)):
        g = pd.Series(m.feature_importance("gain"), index=M.MODEL_FEATURES)
        imp[k] = (g / g.sum()).sort_values(ascending=False).round(4).to_dict()

    # 7. online replay in the test week: materialise as of T, serve requests in (T, T+4min]
    test_trips = use[~is_train]
    cps = S.checkpoint_times(cut + pd.Timedelta(hours=1), test_trips.request_ts.max(), args.checkpoints, cfg.seed)
    online, rep = S.replay_online(store, test_trips, cps)
    skew = S.skew_report(online, pit)
    log(f"online replay: {rep} skew mismatched cells {skew['mismatched_cells']}/{skew['cells']}", t0)
    online_ent = online.drop(columns=[c for c in ("distance_km",) if c in online]).merge(use[["trip_id", "distance_km"]], on="trip_id", how="left")
    y_on = y.loc[online.trip_id].to_numpy()
    pred_on_pit = M.predict(m_pit, online_ent)
    pred_off_pit = M.predict(m_pit, pit.set_index("trip_id").loc[online.trip_id].reset_index())
    replay = {
        **rep,
        "requests": int(len(online)),
        "pit_model_online": M.metrics(y_on, pred_on_pit),
        "naive_model_online": M.metrics(y_on, M.predict(m_naive, online_ent)),
        "max_abs_prediction_diff_online_vs_offline_min": float(np.max(np.abs(pred_on_pit - pred_off_pit))),
    }
    # contrast: a hand-rolled serving path that recomputes traffic from the raw stream
    ent_on = J.entity_frame(use.set_index("trip_id").loc[online.trip_id].reset_index())
    hr = S.handrolled_speed(tables["probes"], ent_on)
    hand = pit.set_index("trip_id").loc[online.trip_id].reset_index()
    rel = np.abs(hr - hand.avg_speed_15m.to_numpy()) / hand.avg_speed_15m.to_numpy()
    hand["avg_speed_15m"] = hr
    hand = J.apply_odfv(hand)
    replay["handrolled_contrast"] = {
        "feature": "avg_speed_15m",
        "rows_differing_gt_1pct_pct": float(np.nanmean(rel > 0.01) * 100.0),
        "median_rel_diff_pct": float(np.nanmedian(rel) * 100.0),
        "pit_model_mae_min": M.metrics(y_on, M.predict(m_pit, hand))["mae_min"],
    }

    # 8. full materialisation + latency benchmark + push + freshness
    tm = time.perf_counter()
    end_all = t_start + pd.Timedelta(days=cfg.n_days)  # "now" = end of the simulated month
    store.materialize(start_date=(end_all - pd.Timedelta(hours=6)).to_pydatetime(), end_date=end_all.to_pydatetime())
    full_mat_s = time.perf_counter() - tm
    lat = S.latency_benchmark(store, J.entity_frame(test_trips), seed=cfg.seed)
    lat_arr = lat.pop("latencies_ms")
    push = S.push_demo(store, int(test_trips.zone_id.iloc[0]), end_all + pd.Timedelta(minutes=5), 7.5)
    ref_test = ref[ref.trip_id.isin(test_trips.trip_id)]
    fresh = {
        "zone_feature_age_min_p50": float(ref_test.zone_feature_age_min.median()),
        "zone_feature_age_min_p99": float(ref_test.zone_feature_age_min.quantile(0.99)),
        "driver_feature_age_min_p50": float(ref_test.driver_feature_age_min.median()),
        "driver_feature_age_min_p99": float(ref_test.driver_feature_age_min.quantile(0.99)),
        "sla_zone_le_10min_pct": float((ref_test.zone_feature_age_min <= 10).mean() * 100.0),
        "sla_driver_le_65min_pct": float((ref_test.driver_feature_age_min <= 65).mean() * 100.0),
    }
    log(f"latency {lat['single_row']}", t0)

    # 9. outputs
    metrics = {
        "data": {
            **prov,
            "trips_used": int(len(use)),
            "train_trips": int(is_train.sum()),
            "test_trips": int((~is_train).sum()),
            "train_window": [str(warm), str(cut)],
            "test_window": [str(cut), str(t_start + pd.Timedelta(days=cfg.n_days))],
        },
        "feature_tables": {k: int(len(v)) for k, v in feats.items()},
        "timings_s": {"feature_build_duckdb": feat_build_s, "naive_join": naive_s, "feast_pit_join": pit_s, "full_materialize": full_mat_s},
        "pit_correctness": {"feast_vs_duckdb_asof": pit_check},
        "leakage": {
            "naive_zone_feature_lookahead_min_mean": float(zone_lookahead.mean()),
            "naive_driver_profile_lookahead_h_mean": float(driver_lookahead_h.mean()),
            "share_trips_finishing_inside_naive_window_pct": float(trip_in_window * 100.0),
            "naive_vs_pit_mismatched_cells_pct": naive_vs_pit["mismatched_cells"] / naive_vs_pit["cells"] * 100.0,
            "naive_vs_pit_per_column": naive_vs_pit["per_column"],
        },
        "models": res,
        "feature_importance_gain_share": {k: dict(list(v.items())[:8]) for k, v in imp.items()},
        "online_replay": replay,
        "skew_test": skew,
        "online_latency": lat,
        "push": push,
        "freshness": fresh,
        "elapsed_s": time.perf_counter() - t0,
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    rows = []
    for k, lab in (("naive", "naive hour-bucket join"), ("pit", "Feast point-in-time join")):
        for stage in ("offline", "production"):
            rows.append({"training_set": lab, "evaluated_on": stage, **res[k][stage]})
    pd.DataFrame(rows).to_csv(paths.results / "model_comparison.csv", index=False)
    online[["trip_id", "checkpoint", *J.ALL_FEATURES]].head(300).to_csv(paths.results / "skew_sample_online_features.csv", index=False, float_format="%.6f")
    pd.DataFrame({"latency_ms": lat_arr}).describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(paths.results / "latency_summary.csv")

    P.offline_vs_production(res, paths.figures / "offline_vs_production_mae.png")
    P.latency_cdf(lat_arr, lat["single_row"]["p50_ms"], lat["single_row"]["p99_ms"], paths.figures / "online_latency_cdf.png")
    # timeline: the busiest zone on the first weekday of the test week
    day = cut.normalize()
    while day.dayofweek >= 5:
        day += pd.Timedelta(days=1)
    z = int(trips.zone_id.value_counts().idxmax())
    P.leakage_timeline(feats["zone_traffic"], snaps["zone_traffic_hourly"], trips, z, day, paths.figures / "leakage_timeline.png")
    write_report(metrics, paths.results / "report.md")
    log("done", t0)


def write_report(m: dict, out: Path) -> None:
    r = m["models"]
    lat = m["online_latency"]["single_row"]
    sk = m["skew_test"]
    pc = m["pit_correctness"]["feast_vs_duckdb_asof"]
    lines = [
        "# Day 17 run report (generated by scripts/run_pipeline.py)",
        "",
        f"Data source: **{m['data']['source']}** (SYNTHETIC unless stated otherwise). TLC probe: `{m['data']['tlc_status']}`.",
        f"Trips used: {m['data']['trips_used']:,} (train {m['data']['train_trips']:,}, test {m['data']['test_trips']:,}).",
        "",
        "## Offline vs production ETA error (test week, minutes)",
        "",
        "| Training set | Offline MAE | Production MAE | Gap | Gap % | Production MAPE % |",
        "|---|---|---|---|---|---|",
    ]
    for k, lab in (("naive", "Naive hour-bucket join"), ("pit", "Feast point-in-time join")):
        lines.append(
            f"| {lab} | {r[k]['offline']['mae_min']:.3f} | {r[k]['production']['mae_min']:.3f} | {r[k]['gap_mae_min']:+.3f} | {r[k]['gap_pct']:+.1f}% | {r[k]['production']['mape_pct']:.1f} |"
        )
    lines += [
        f"| Heuristic (distance / PIT speed x driver factor) | - | {r['heuristic_driver_adjusted_eta']['mae_min']:.3f} | - | - | {r['heuristic_driver_adjusted_eta']['mape_pct']:.1f} |",
        "",
        f"In production the naive model's MAE is {r['naive_vs_pit_production_mae_pct']:.1f}% higher than the PIT model's.",
        "",
        "## Point-in-time correctness",
        "",
        f"Feast vs independent DuckDB ASOF join: {pc['mismatched_cells']} mismatched cells out of {pc['cells']:,} ({pc['rows']:,} rows).",
        "",
        "## Training/serving skew (online replay)",
        "",
        f"{m['online_replay']['checkpoints']} checkpoints, {m['online_replay']['requests']} requests served from the SQLite online store: "
        f"{sk['mismatched_cells']} mismatched cells out of {sk['cells']:,}; max prediction difference online vs offline "
        f"{m['online_replay']['max_abs_prediction_diff_online_vs_offline_min']:.2e} min.",
        f"Hand-rolled serving contrast: {m['online_replay']['handrolled_contrast']['rows_differing_gt_1pct_pct']:.1f}% of rows differ by >1% on avg_speed_15m; "
        f"PIT model MAE {m['online_replay']['handrolled_contrast']['pit_model_mae_min']:.4f} vs {m['online_replay']['pit_model_online']['mae_min']:.4f} with store features.",
        "",
        "## Online latency (single request, feature service with on-demand transform)",
        "",
        f"p50 {lat['p50_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms, p99 {lat['p99_ms']:.2f} ms, max {lat['max_ms']:.2f} ms over {lat['n']} requests.",
        "",
        "## Freshness",
        "",
        json.dumps(m["freshness"], indent=2),
    ]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
