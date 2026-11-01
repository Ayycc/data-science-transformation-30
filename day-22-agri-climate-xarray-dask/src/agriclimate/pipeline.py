"""End-to-end run: cube -> indices -> zonal stats -> yield model (random vs spatial CV) -> scenario -> perf notes."""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from . import figures, public
from .config import SCENARIO_DT_C, SCENARIO_PRECIP_FACTOR, RunConfig
from .generate import grid, soil_index, write_cube
from .indices import (
    compute_climate_features,
    ndvi_season,
    open_cube,
    perturb,
)
from .model import (
    CLIMATE_FEATURES,
    NOWCAST_FEATURES,
    add_true_yield,
    cross_validate,
    make_model,
    risk_metrics,
    scenario_table,
)
from .perf import MemorySampler, benchmark, chunk_info, run_out_of_core
from .zonal import field_table, fields_to_geojson, label_raster, make_fields, zonal_mean


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_cube(cfg: RunConfig, path: Path, try_public: bool) -> dict:
    """Public ERA5 path if reachable (not reached in the build environment), else the synthetic generator."""
    info: dict = {"public_source": "Open-Meteo ERA5 archive API", "public_attempted": try_public}
    reachable, msg = public.probe() if try_public else (False, "not attempted")
    info["public_reachable"] = reachable
    info["public_probe_message"] = msg
    if reachable:
        try:
            coarse = public.fetch_era5(cfg.cube.years)
            info["path_used"] = "public_era5_open_meteo"
            info |= write_public_cube(cfg, coarse, path)
            return info
        except Exception as exc:  # fall back on any failure of the public path
            info["public_fetch_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    info["path_used"] = "synthetic_generator"
    info |= write_cube(cfg.cube, path, cfg.seed)
    return info


def write_public_cube(cfg: RunConfig, coarse: xr.Dataset, path: Path) -> dict:
    """Interpolate public climate to the grid, derive SYNTHETIC NDVI from it, and write the same Zarr layout."""
    from .generate import N_COMPOSITES, _template, ndvi_from_climate

    lat, lon = grid(cfg.cube)
    clim = public.interpolate_to_grid(coarse, lat, lon)
    tmpl = _template(cfg.cube)
    soil = soil_index(lat, lon, cfg.seed).ravel()
    lo = np.tile(lon, lat.size)
    rng = np.random.default_rng([cfg.seed, 404])
    ndvi = []
    for y in cfg.cube.years:
        c = clim.sel(time=str(y))
        n = c.sizes["time"]
        tx = c["tmax"].values.reshape(n, -1)
        tn = c["tmin"].values.reshape(n, -1)
        pr = c["precip"].values.reshape(n, -1)
        doy = c["time"].dt.day_of_year.values.astype(float)
        ndvi.append(ndvi_from_climate(tx, (tx + tn) / 2, pr, doy, lo, soil, rng).reshape(N_COMPOSITES, lat.size, lon.size))
    out = tmpl.copy()
    for v in ("tmax", "tmin", "precip"):
        out[v] = (("time", "lat", "lon"), clim[v].values.astype("float32"))
    out["ndvi"] = (("time_ndvi", "lat", "lon"), np.concatenate(ndvi).astype("float32"))
    out.attrs["title"] = "ERA5 (Open-Meteo) climate + SYNTHETIC NDVI"
    out = out.chunk({"time": cfg.cube.chunk_time, "lat": cfg.cube.chunk_lat, "lon": cfg.cube.chunk_lon})
    out.to_zarr(path, mode="w", consolidated=True)
    return {"path": str(path), "uncompressed_bytes": int(out.nbytes)}


def field_year_table(feats: xr.Dataset, labels: xr.DataArray, fields_df: pd.DataFrame, extra: xr.Dataset | None) -> pd.DataFrame:
    data = feats if extra is None else xr.merge([feats, extra])
    df = zonal_mean(data, labels)
    return df.merge(fields_df[["field_id", "lat", "lon", "n_cells"]], on="field_id")


SAMPLE_SLICE = {"lat": slice(0, 8), "lon": slice(0, 8)}


def write_cube_sample(ds: xr.Dataset, path: Path) -> None:
    """Small committed cut of the cube (8 x 8 cells, year 2012) for offline inspection and a determinism test."""
    sub = ds.isel(SAMPLE_SLICE).sel(time="2012", time_ndvi="2012").load()
    sub.attrs = {k: str(v) for k, v in sub.attrs.items()}
    for v in sub.variables:
        sub[v].encoding = {}
    sub.to_netcdf(path, engine="scipy")


def run(cfg: RunConfig, root: Path, try_public: bool = True, out_of_core: bool = True, keep_big: bool = False) -> dict:
    root = Path(root)
    data_dir, res_dir, fig_dir = root / "data", root / "results", root / "results" / "figures"
    for d in (data_dir, res_dir, fig_dir, data_dir / "sample"):
        d.mkdir(parents=True, exist_ok=True)
    zarr_path = data_dir / "cube.zarr"
    metrics: dict = {
        "config": {"seed": cfg.seed, "python": platform.python_version(), "dask_scheduler": "threads (local)", "num_workers": cfg.num_workers}
    }
    t_all = time.perf_counter()
    dask.config.set(scheduler="threads", num_workers=cfg.num_workers)

    with MemorySampler(0.1) as ms_main:
        _log("1/8 building the data cube")
        t0 = time.perf_counter()
        info = build_cube(cfg, zarr_path, try_public)
        info["write_wall_s"] = time.perf_counter() - t0
        ds = open_cube(str(zarr_path))
        info["dims"] = {k: int(v) for k, v in ds.sizes.items()}
        info["cube_gb_uncompressed"] = float(ds.nbytes / 2**30)
        info["disk_mb"] = float(sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()) / 2**20)
        info["chunks"] = {v: chunk_info(ds[v]) for v in ds.data_vars}
        info["synthetic_labels"] = (
            "climate, NDVI, fields and yields are SYNTHETIC"
            if info["path_used"] == "synthetic_generator"
            else "NDVI, fields and yields are SYNTHETIC"
        )
        metrics["data"] = info
        write_cube_sample(ds, data_dir / "sample" / "cube_sample_2012_SYNTHETIC.nc")

        _log("2/8 climate indices (GDD, heat days, JJA precip, SPI-3) with Dask")
        t0 = time.perf_counter()
        feats, params, tasks = compute_climate_features(ds, num_workers=cfg.num_workers)
        nd = ndvi_season(ds).compute()
        metrics["indices"] = {"wall_s": time.perf_counter() - t0, "graph_tasks_per_feature": tasks}

        _log("3/8 fields + zonal statistics")
        fields = make_fields(cfg.n_fields, cfg.seed)
        fields_to_geojson(fields, data_dir / "sample" / "fields.geojson")
        labels, fallback = label_raster(fields, ds["tmax"].isel(time=0))
        fdf = field_table(fields, labels)
        lat, lon = grid(cfg.cube)
        soil = xr.DataArray(soil_index(lat, lon, cfg.seed), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="soil")
        soil_y = soil.expand_dims(year=feats["year"].values)
        table = field_year_table(xr.merge([feats, nd]), labels, fdf, soil_y.to_dataset())
        table = add_true_yield(table, cfg.seed)
        metrics["zonal"] = {
            "n_fields": len(fields),
            "n_field_years": len(table),
            "cells_per_field_median": float(fdf["n_cells"].median()),
            "cells_per_field_min": int(fdf["n_cells"].min()),
            "centroid_fallback_fields": int(fallback),
        }
        sample_cols = ["field_id", "year", "lat", "lon", "gdd", "heat_days", "precip_jja", "spi3_aug", "ndvi_int", "ndvi_peak", "yield_t_ha"]
        table[sample_cols].round(3).to_csv(data_dir / "sample" / "field_year_yields_SYNTHETIC.csv", index=False)

        _log("4/8 yield models: random vs spatially blocked CV")
        cv = {}
        for fs_name, fs in (("nowcast_with_ndvi", NOWCAST_FEATURES), ("climate_only", CLIMATE_FEATURES)):
            for scheme in ("random", "spatial"):
                cv[f"{fs_name}__{scheme}"] = cross_validate(table, fs, scheme, cfg.n_folds, cfg.block_deg, cfg.seed)
        y = table["yield_t_ha"].to_numpy()
        cv["baseline_mean"] = {"rmse": float(np.sqrt(np.mean((y - y.mean()) ** 2)))}
        for fs_name in ("nowcast_with_ndvi", "climate_only"):
            r, s = cv[f"{fs_name}__random"], cv[f"{fs_name}__spatial"]
            cv[f"{fs_name}__rmse_optimism_pct"] = 100 * (s["rmse"] / r["rmse"] - 1)
        cv["block_deg"] = cfg.block_deg
        metrics["cv"] = cv

        _log("5/8 climate scenario: +2 C, -20% precipitation (lazy perturbation of the cube)")
        t0 = time.perf_counter()
        ds_s = perturb(ds, SCENARIO_DT_C, SCENARIO_PRECIP_FACTOR)
        feats_s, _, _ = compute_climate_features(ds_s, params=params, num_workers=cfg.num_workers)
        table_s = field_year_table(feats_s, labels, fdf, None)
        model = make_model(cfg.seed).fit(table[CLIMATE_FEATURES].to_numpy(), table["yield_t_ha"].to_numpy())
        ref_year = int(max(cfg.cube.years))
        scen = scenario_table(table, table_s, model, CLIMATE_FEATURES, ref_year)
        risk_model = risk_metrics(scen, "pred_base", "pred_scen")
        risk_true = risk_metrics(scen, "true_base", "true_scen")
        out_of_range = {
            f: float((table_s[f] > table[f].max()).mean() + (table_s[f] < table[f].min()).mean())
            for f in ("gdd", "heat_days", "precip_jja", "spi3_aug")
        }
        per_field = scen.groupby("field_id").agg(
            base_mean=("pred_base", "mean"), scen_mean=("pred_scen", "mean"), scen_p5=("pred_scen", lambda v: np.percentile(v, 5))
        )
        per_field["change_pct"] = 100 * (per_field["scen_mean"] / per_field["base_mean"] - 1)
        per_field["yar95_scen"] = per_field["scen_mean"] - per_field["scen_p5"]
        per_field = per_field.join(fdf.set_index("field_id")[["lat", "lon"]])
        per_field.round(4).to_csv(res_dir / "field_risk.csv")
        metrics["scenario"] = {
            "dt_c": SCENARIO_DT_C,
            "precip_factor": SCENARIO_PRECIP_FACTOR,
            "reference_technology_year": ref_year,
            "n_field_weather_years": len(scen),
            "model": risk_model,
            "truth": risk_true,
            "model_vs_truth_mean_change_gap_pp": risk_model["mean_change_pct"] - risk_true["mean_change_pct"],
            "share_scenario_rows_outside_training_range": out_of_range,
            "mean_feature_shift": {f: float(table_s[f].mean() - table[f].mean()) for f in ("gdd", "heat_days", "precip_jja", "spi3_aug")},
            "fields_with_mean_loss_over_20pct": int((per_field["change_pct"] < -20).sum()),
            "wall_s": time.perf_counter() - t0,
        }
        metrics["main_pipeline_peak_rss_mb"] = ms_main.peak_mb

    _log("6/8 performance notes: chunk sizes, task graphs, memory")
    bench_dir = data_dir / "ooc"
    rows = benchmark(cfg.cube, zarr_path, bench_dir, cfg.num_workers, cfg.seed, [("tiny", "dot"), ("stored", "dot"), ("large", "dot")])
    names = {"tiny": "tiny", "stored": "stored", "large": "large"}
    for r in rows:
        r["name"] = f"{names[r['chunking']]} {tuple(r['chunk_shape'])}"
    perf = {"chunk_experiment": rows}

    if out_of_core:
        _log("7/8 out-of-core demonstration (child processes with a hard RLIMIT_AS memory cap)")
        big_path = data_dir / "big_cube.zarr"
        ooc = run_out_of_core(cfg.big_cube, big_path, data_dir / "ooc", cfg.big_mem_cap_gb, cfg.num_workers, cfg.seed)
        traces = {k: ooc[k].pop("rss_trace") for k in ("write", "load", "compute")}
        perf["out_of_core"] = ooc
        rd = benchmark(cfg.big_cube, big_path, bench_dir, cfg.num_workers, cfg.seed, [("stored", "resample"), ("stored", "dot")])
        perf["resample_vs_dot_big_cube"] = {r["method"]: r for r in rd}
        perf["resample_vs_dot_big_cube"]["same_result"] = bool(abs(rd[0]["checksum"] - rd[1]["checksum"]) <= 1e-6 * abs(rd[1]["checksum"]))
        figures.memory_profile(traces, cfg.big_mem_cap_gb, ooc["summary"], fig_dir / "out_of_core_memory.png")
        if not keep_big:
            shutil.rmtree(big_path, ignore_errors=True)
            perf["out_of_core"]["big_cube_deleted_after_run"] = True
    metrics["performance"] = perf

    _log("8/8 figures and reports")
    figures.drought_map(feats, nd, fields, 2012, fig_dir / "drought_2012_spi_ndvi.png")
    figures.cv_comparison(cv, fig_dir / "cv_random_vs_spatial.png")
    figures.scenario_distribution(scen, risk_model, risk_true, fig_dir / "scenario_yield_distribution.png")
    figures.chunking(perf["chunk_experiment"], fig_dir / "chunking_experiment.png")
    figures.exposure_map_png(per_field, fields, fig_dir / "field_exposure_map.png")
    figures.exposure_map_html(per_field, fields, res_dir / "drought_exposure_map.html")
    metrics["total_wall_s"] = time.perf_counter() - t_all
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    from .report import write_reports

    write_reports(metrics, res_dir)
    _log(f"done in {metrics['total_wall_s']:.0f} s")
    return metrics
