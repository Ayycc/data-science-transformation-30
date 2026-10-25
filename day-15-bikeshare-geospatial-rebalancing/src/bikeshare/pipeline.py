"""End-to-end run: data -> spatial features -> forecasts -> routing -> simulation -> results/."""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from . import band, data, geo, maps, routing
from . import config as C
from . import features as F
from . import forecast as FC
from . import simulate as SIM


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def run(force_synthetic: bool = False, out_dir=C.RESULTS) -> dict:
    t0 = time.time()
    cfg = C.FLEET
    figs, mapdir = out_dir / "figures", out_dir / "maps"
    figs.mkdir(parents=True, exist_ok=True)
    mapdir.mkdir(parents=True, exist_ok=True)

    # 1. data -------------------------------------------------------------------------------------
    acq = data.acquire(force_synthetic=force_synthetic)
    _log(f"data: {acq.provenance['n_trips']:,} trips, {acq.provenance['n_stations']} stations ({acq.provenance['trips']['source']})", t0)

    # 2. spatial features -------------------------------------------------------------------------
    stations, geo_prov = geo.add_spatial_features(acq.stations)
    if not acq.provenance["trips"]["synthetic"]:
        data.write_samples(acq.trips, stations, acq.weather)
    nbrs = geo.neighbour_lists(stations)
    _log(f"spatial features: {stations['h3'].nunique()} H3 cells; {geo_prov}", t0)

    # 3. panel + forecasts ------------------------------------------------------------------------
    dep, arr, hours = F.hourly_counts(acq.trips, stations["station_id"], C.TRAIN_START, C.TEST_END)
    panel = F.build_panel(dep, arr, hours, stations, acq.weather, nbrs)
    tm = panel["time"]
    fit = ((tm >= C.TRAIN_START + pd.Timedelta(days=7)) & (tm < C.VALID_START)).to_numpy()
    valid = ((tm >= C.VALID_START) & (tm < C.TEST_START)).to_numpy()
    train = (tm < C.TEST_START).to_numpy()
    test = (tm >= C.TEST_START).to_numpy()
    fc = FC.fit_lgbm(panel, fit, valid)
    preds = {
        "lgbm": fc.predict(panel),
        "historical_average": FC.historical_average(panel, train).to_numpy(),
        "seasonal_naive_168h": panel["lag168"].fillna(0).to_numpy(),
        "zero": np.zeros(len(panel)),
    }
    ev = FC.evaluate(panel, test, preds)
    per_station = ev.pop("per_station_mae")
    hz = FC.horizon_error(panel, test, preds)
    _log(f"forecast: test MAE lgbm {ev['models']['lgbm']['mae']} vs HA {ev['models']['historical_average']['mae']}", t0)

    # 3b. decision-focused band forecast (drain / fill per station-shift) ---------------------------
    net_grid = panel["net"].to_numpy().reshape(len(stations), -1)
    table = band.band_table(net_grid, hours, acq.weather, stations, nbrs)
    bd = table["date"]
    b_fit = ((bd >= C.TRAIN_START + pd.Timedelta(days=7)) & (bd < C.VALID_START)).to_numpy()
    b_valid = ((bd >= C.VALID_START) & (bd < C.TEST_START)).to_numpy()
    b_test = (bd >= C.TEST_START).to_numpy()
    bmodel = band.fit_band(table, b_fit, b_valid)
    b_pred = bmodel.predict(table)
    b_ha = band.band_baseline(table, (bd < C.TEST_START).to_numpy())
    band_eval = {
        t: {
            "mae_lgbm": round(FC.mae(b_pred[t][b_test], table[t][b_test]), 4),
            "mae_historical_average": round(FC.mae(b_ha[t][b_test], table[t][b_test]), 4),
            "mae_hourly_lgbm_path": None,
            "mean_actual": round(float(table[t][b_test].mean()), 3),
        }
        for t in band.TARGETS
    }
    _log(f"band forecast: {band_eval}", t0)

    # 4. routing + simulation ---------------------------------------------------------------------
    S = len(stations)
    T_test = int(test.sum() // S)
    to_grid = lambda p: p[test].reshape(S, T_test)  # noqa: E731 - panel is station-major
    # how well the extremes of the hourly LightGBM path estimate the band (the Jensen gap)
    hp = to_grid(preds["lgbm"])
    for t_i, t in enumerate(band.TARGETS):
        est = []
        rows = table.loc[b_test]
        h0 = ((rows["date"] - C.TEST_START).dt.days * 24 + rows["shift_id"].map({k: s[1] for k, s in enumerate(cfg.shifts)})).to_numpy()
        hl = rows["shift_id"].map({k: s[3] - s[1] for k, s in enumerate(cfg.shifts)}).to_numpy()
        st_i = rows["station"].to_numpy()
        for L in np.unique(hl):
            m_ = hl == L
            idx = h0[m_][:, None] + np.arange(L)[None, :]
            d_, f_ = SIM.path_band(hp[st_i[m_][:, None], idx])
            est.append(pd.Series(d_ if t_i == 0 else f_, index=rows.index[m_]))
        est_s = pd.concat(est).reindex(rows.index)
        band_eval[t]["mae_hourly_lgbm_path"] = round(FC.mae(est_s, rows[t]), 4)
        band_eval[t]["mean_hourly_lgbm_path"] = round(float(est_s.mean()), 3)

    depot_xy = stations[["lat", "lon"]].to_numpy()
    w = stations["activity"].to_numpy(float)
    depot = (float(np.average(depot_xy[:, 0], weights=w)), float(np.average(depot_xy[:, 1], weights=w)))
    tr = panel.loc[train & (panel["offday"] == 0).to_numpy()]
    weekday_profile = tr.groupby(["station", "hour"])["net"].mean().unstack().reindex(range(S)).fillna(0).to_numpy()

    # 4a. policy selection on the validation week, with models refit on data before it
    selection = select_policy(panel, table, acq, stations, depot, cfg, t0)

    # 4b. test fortnight
    fixed = SIM.FixedRoutePolicy.build(stations, weekday_profile, depot, cfg)
    actual_band = table[list(band.TARGETS)]
    bands = {
        "dynamic_ha_band": SIM.table_band_fn(table[b_test], b_ha[b_test], S),
        "dynamic_lgbm_hourly": SIM.hourly_band_fn(hp, C.TEST_START),
        "dynamic_lgbm_band": SIM.table_band_fn(table[b_test], b_pred[b_test], S),
        "dynamic_oracle": SIM.table_band_fn(table[b_test], actual_band[b_test], S),
    }
    policies: list[SIM.Policy] = [SIM.Policy(), fixed] + [
        SIM.ForecastPolicy(name=k, stations=stations, band_fn=f, depot=depot, cfg=cfg) for k, f in bands.items()
    ]
    sims: dict[str, SIM.SimResult] = {}
    for pol in policies:
        sims[pol.name] = SIM.simulate(stations, acq.trips, pol, C.TEST_START, C.TEST_END, cfg)
        _log(f"simulated {pol.name}: {sims[pol.name].summary()}", t0)

    summary = {k: v.summary() for k, v in sims.items()}
    base_none = summary["none"]["stockout_hours"]
    base_fixed = summary["fixed_routes"]["stockout_hours"]
    for s in summary.values():
        s["stockout_reduction_vs_none_pct"] = round(100 * (1 - s["stockout_hours"] / base_none), 2)
        s["stockout_reduction_vs_fixed_pct"] = round(100 * (1 - s["stockout_hours"] / base_fixed), 2)
        s["lost_departures_reduction_vs_fixed_pct"] = round(
            100 * (1 - s["lost_departures"] / max(1, summary["fixed_routes"]["lost_departures"])), 2
        )

    # routing constraint audit over every plan the simulation executed
    audit = {}
    shift_len = {s[0]: s[2] for s in cfg.shifts}
    for pol in policies:
        if isinstance(pol, SIM.ForecastPolicy):
            df = pd.DataFrame(pol.checks)
        elif isinstance(pol, SIM.FixedRoutePolicy):
            df = pd.DataFrame(
                [{"shift": k, **routing.check_plan(p, cfg, shift_len[k] * 60), "km": p.distance_m / 1000} for k, p in pol.plans.items()]
            )
        else:
            continue
        audit[pol.name] = {
            "plans": int(len(df)),
            "all_feasible": bool(df["feasible"].all()),
            "max_truck_load": int(df["max_load"].max()),
            "truck_capacity": cfg.truck_capacity,
            "max_route_minutes": {s: round(float(g["max_duration_s"].max() / 60), 1) for s, g in df.groupby("shift")},
            "shift_minutes": shift_len,
            "mean_km_per_shift": round(float(df["km"].mean()), 1),
        }

    # 5. maps + figures ---------------------------------------------------------------------------
    am = panel.loc[train & (panel["offday"] == 0).to_numpy() & panel["hour"].between(7, 9).to_numpy()]
    am_net = am.groupby("station")["net"].mean().reindex(range(S)).fillna(0)
    cells = geo.h3_cells_gdf(stations, pd.DataFrame({"am_net": am_net.to_numpy()}))
    cells["am_net_per_station"] = cells["am_net"] / cells["n_stations"]
    maps.feature_map(stations, cells, am_net, mapdir / "station_h3_features.html")
    demo_day = pd.Timestamp("2024-09-17")  # first Tuesday of the test period
    lg = next(p for p in policies if p.name == "dynamic_lgbm_band")
    demo_idx = [i for i, c in enumerate(lg.checks) if c["time"].startswith(str(demo_day.date()))]
    for i in demo_idx:
        plan = sims["dynamic_lgbm_band"].plans[i]
        label = f"LightGBM-driven {lg.checks[i]['shift']} routes, {demo_day.date()} ({sum(map(len, plan.routes))} stops, {plan.planned_bikes} bikes planned)"
        maps.routes_map(stations, plan, depot, label, mapdir / f"routes_{lg.checks[i]['shift']}_{demo_day.date()}.html")
        if lg.checks[i]["shift"] == "overnight":
            maps.fig_routes_static(stations, cells, plan, depot, label, figs / "routes_overnight.png")
    maps.before_after_map(
        stations, {k: sims[k] for k in ("none", "fixed_routes", "dynamic_lgbm_band")}, mapdir / "stockouts_before_after.html"
    )
    cells.drop(columns="geometry").assign(geometry=cells.geometry.to_wkt()).to_csv(out_dir / "h3_cells.csv", index=False)

    busiest = int(np.argmax(stations["activity"].to_numpy()))
    rows = panel.loc[test & (panel["station"] == busiest).to_numpy()]
    first_week = rows["time"] < C.TEST_START + pd.Timedelta(days=7)
    maps.fig_forecast(
        pd.DatetimeIndex(rows.loc[first_week, "time"]),
        rows.loc[first_week, "net"].to_numpy(),
        {"LightGBM": preds["lgbm"][rows.index[first_week]], "historical average": preds["historical_average"][rows.index[first_week]]},
        f"Hourly net flow, {stations.at[busiest, 'name']} (test week 1)",
        figs / "forecast_busiest_station.png",
    )
    names = list(ev["models"])
    maps.fig_bars(
        names,
        [ev["models"][n]["mae"] for n in names],
        [maps.BLUE, maps.ORANGE, "#c3c2b7", "#c3c2b7"],
        "MAE, bikes per station-hour",
        "Test-period station-hour MAE (lower is better)",
        figs / "forecast_mae.png",
    )
    maps.fig_stockouts(summary, figs / "stockouts_by_policy.png")
    maps.fig_importance(fc.importance, figs / "feature_importance.png")

    st_out = stations[
        [
            "station_id",
            "name",
            "lat",
            "lon",
            "h3",
            "capacity",
            "is_metro",
            "metro_dist_m",
            "elevation_m",
            "elev_rel_ring2",
            "stations_ring2",
        ]
    ].copy()
    st_out["mae_lgbm"] = per_station["lgbm"].reindex(range(S)).round(4).to_numpy()
    st_out["mae_ha"] = per_station["historical_average"].reindex(range(S)).round(4).to_numpy()
    for k in ("none", "fixed_routes", "dynamic_lgbm_band"):
        st_out[f"stockout_h_{k}"] = np.round((sims[k].empty_min + sims[k].full_min) / 60, 2)
    st_out.to_csv(out_dir / "stations.csv", index=False)

    metrics = {
        "data": {**acq.provenance, "spatial": geo_prov, "h3_resolution": C.H3_RES, "h3_cells": int(stations["h3"].nunique())},
        "split": {
            "train": f"{C.TRAIN_START.date()}..{(C.TEST_START - pd.Timedelta(days=1)).date()}",
            "validation_week": str(C.VALID_START.date()),
            "test": f"{C.TEST_START.date()}..{(C.TEST_END - pd.Timedelta(days=1)).date()}",
        },
        "forecast": {
            **ev,
            "horizon_cum_mae": hz,
            "lgbm": {"objective": fc.objective, "best_iteration": fc.best_iteration, "selection": fc.validation},
            "top_features": fc.importance.head(10).round(4).to_dict(),
        },
        "band_forecast": {**band_eval, "iterations": bmodel.iterations},
        "policy_selection_validation_week": selection,
        "fleet": {k: v for k, v in cfg.__dict__.items()},
        "depot": depot,
        "simulation": summary,
        "routing_audit": audit,
        "runtime_s": round(time.time() - t0, 1),
        "env": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("lightgbm", "ortools", "geopandas", "h3", "osmnx", "folium", "pandas", "numpy")},
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _write_report(metrics, out_dir / "report.md")
    _log("done", t0)
    return metrics


def _write_report(m: dict, path) -> None:
    f = m["forecast"]["models"]
    lines = [
        "# Day 15 results report",
        "",
        f"Data: {m['data']['trips']['source']}. Generated by `make run`.",
        "",
        "## Forecast (test period, station-hour net flow)",
        "",
    ]
    lines += ["| model | MAE | RMSE | MAE weekday peaks |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mae']} | {v['rmse']} | {v['mae_weekday_peaks']} |" for k, v in f.items()]
    lines += [
        "",
        "Cumulative-horizon MAE (bikes per station-shift):",
        "",
        "| model | " + " | ".join(m["forecast"]["horizon_cum_mae"]) + " |",
        "|---|" + "---|" * len(m["forecast"]["horizon_cum_mae"]),
    ]
    for k in f:
        lines.append(f"| {k} | " + " | ".join(str(m["forecast"]["horizon_cum_mae"][s][k]) for s in m["forecast"]["horizon_cum_mae"]) + " |")
    lines += [
        "",
        "## Simulation (14-day test)",
        "",
        "| policy | stockout h | empty h | full h | lost trips | diverted | bikes moved | truck km | vs none | vs fixed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, s in m["simulation"].items():
        lines.append(
            f"| {k} | {s['stockout_hours']:,} | {s['empty_dock_hours']:,} | {s['full_dock_hours']:,} | {s['lost_departures']:,} | {s['diverted_arrivals']:,} | {s['bikes_moved']:,} | {s['truck_km']:,} | {s['stockout_reduction_vs_none_pct']}% | {s['stockout_reduction_vs_fixed_pct']}% |"
        )
    lines += [
        "",
        "## Band forecast (bikes per station-shift, test)",
        "",
        "| target | LightGBM band | historical average | extremes of hourly LightGBM path | mean actual |",
        "|---|---|---|---|---|",
    ]
    for t in ("drain", "fill"):
        b = m["band_forecast"][t]
        lines.append(f"| {t} | {b['mae_lgbm']} | {b['mae_historical_average']} | {b['mae_hourly_lgbm_path']} | {b['mean_actual']} |")
    sel = m["policy_selection_validation_week"]
    lines += ["", f"## Policy selection on the validation week ({sel['window']})", "", "| candidate | stockout h |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in sel["stockout_hours"].items()]
    lines += ["", f"Selected: `{sel['selected']}`."]
    lines += [
        "",
        "## Routing audit",
        "",
        "| policy | plans | all feasible | max load / cap | max route min (shift min) | mean km/shift |",
        "|---|---|---|---|---|---|",
    ]
    for k, a in m["routing_audit"].items():
        lines.append(
            f"| {k} | {a['plans']} | {a['all_feasible']} | {a['max_truck_load']} / {a['truck_capacity']} | {a['max_route_minutes']} ({a['shift_minutes']}) | {a['mean_km_per_shift']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def select_policy(panel: pd.DataFrame, table: pd.DataFrame, acq, stations: pd.DataFrame, depot, cfg, t0: float) -> dict:
    """Choose the forecast -> band rule on the validation week (9-15 Sep) before touching the test set.

    Models are refit on 8 Aug - 1 Sep with 2-8 Sep for early stopping, then each candidate policy is
    simulated on 9-15 Sep. Only the ranking is used; the test numbers come from the final models.
    """
    v_start, e_start, e_end = C.VALID_START - pd.Timedelta(days=7), C.VALID_START, C.TEST_START
    S = len(stations)
    tm = panel["time"]
    fit = ((tm >= C.TRAIN_START + pd.Timedelta(days=7)) & (tm < v_start)).to_numpy()
    val = ((tm >= v_start) & (tm < e_start)).to_numpy()
    ev = ((tm >= e_start) & (tm < e_end)).to_numpy()
    fc = FC.fit_lgbm(panel, fit, val, objectives=("regression",))
    T = int(ev.sum() // S)
    hp = fc.predict(panel)[ev].reshape(S, T)
    ha = FC.historical_average(panel, (tm < e_start).to_numpy()).to_numpy()[ev].reshape(S, T)
    bd = table["date"]
    bf = ((bd >= C.TRAIN_START + pd.Timedelta(days=7)) & (bd < v_start)).to_numpy()
    bv = ((bd >= v_start) & (bd < e_start)).to_numpy()
    be = ((bd >= e_start) & (bd < e_end)).to_numpy()
    bp = band.fit_band(table, bf, bv).predict(table)
    bb = band.band_baseline(table, (bd < e_start).to_numpy())
    cands = {
        "ha_hourly": SIM.hourly_band_fn(ha, e_start),
        "lgbm_hourly": SIM.hourly_band_fn(hp, e_start),
        "ha_band": SIM.table_band_fn(table[be], bb[be], S),
        "lgbm_band": SIM.table_band_fn(table[be], bp[be], S),
    }
    out = {}
    for name, fn in cands.items():
        pol = SIM.ForecastPolicy(name=name, stations=stations, band_fn=fn, depot=depot, cfg=cfg)
        out[name] = SIM.simulate(stations, acq.trips, pol, e_start, e_end, cfg).summary()["stockout_hours"]
        _log(f"selection week {name}: {out[name]} stockout h", t0)
    return {
        "window": f"{e_start.date()}..{(e_end - pd.Timedelta(days=1)).date()}",
        "stockout_hours": out,
        "selected": min(out, key=out.get),
    }
