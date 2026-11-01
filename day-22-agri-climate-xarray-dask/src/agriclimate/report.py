"""Markdown reports generated from metrics.json (so every quoted number comes from the run)."""

from __future__ import annotations

from pathlib import Path


def _cv_rows(cv: dict) -> list[str]:
    rows = ["| Model | CV scheme | RMSE (t/ha) | MAE (t/ha) | R2 |", "|---|---|---|---|---|"]
    for fs in ("nowcast_with_ndvi", "climate_only"):
        for scheme in ("random", "spatial"):
            r = cv[f"{fs}__{scheme}"]
            rows.append(f"| {fs} | {scheme} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |")
    rows.append(f"| predict the mean | - | {cv['baseline_mean']['rmse']:.3f} | - | 0 |")
    return rows


def _risk_rows(sc: dict) -> list[str]:
    rows = [
        "| Source | Climate | Mean (t/ha) | P5 (t/ha) | YaR95 = mean - P5 (t/ha) | CVaR95 (t/ha) | P(yield < 80% of field baseline mean) |",
        "|---|---|---|---|---|---|---|",
    ]
    for src in ("model", "truth"):
        for clim in ("baseline", "scenario"):
            r = sc[src][clim]
            rows.append(
                f"| {src} | {clim} | {r['mean_t_ha']:.2f} | {r['p5_t_ha']:.2f} | {r['yar95_t_ha']:.2f} | {r['cvar95_t_ha']:.2f} | {100 * r['p_below_trigger']:.1f}% |"
            )
    return rows


def write_reports(m: dict, res_dir: Path) -> None:
    cv, sc, perf, d = m["cv"], m["scenario"], m["performance"], m["data"]
    lines = [
        "# Day 22 run report (SYNTHETIC data)",
        "",
        f"Data path used: **{d['path_used']}** (public probe: {d['public_probe_message']}).",
        f"Cube: {d['dims']} = {d['cube_gb_uncompressed']:.2f} GB uncompressed, {d['disk_mb']:.0f} MB in Zarr. {m['zonal']['n_fields']} fields, {m['zonal']['n_field_years']} field-years.",
        "",
        "## Yield model cross-validation",
        "",
        *_cv_rows(cv),
        "",
        f"Spatial-block RMSE is {cv['nowcast_with_ndvi__rmse_optimism_pct']:.1f}% higher than random-CV RMSE for the nowcast model "
        f"and {cv['climate_only__rmse_optimism_pct']:.1f}% higher for the climate-only model ({cv['block_deg']} degree blocks).",
        "",
        f"## Climate scenario: +{sc['dt_c']} C, precipitation x{sc['precip_factor']}",
        "",
        *_risk_rows(sc),
        "",
        f"Model mean change {sc['model']['mean_change_pct']:+.1f}% vs truth {sc['truth']['mean_change_pct']:+.1f}% "
        f"(gap {sc['model_vs_truth_mean_change_gap_pp']:+.1f} pp). Fields with a mean loss over 20%: {sc['fields_with_mean_loss_over_20pct']}.",
        "",
        "Mean feature shift under the scenario: " + ", ".join(f"{k} {v:+.1f}" for k, v in sc["mean_feature_shift"].items()) + ".",
        "",
        "Share of scenario field-years outside the training range: "
        + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sc["share_scenario_rows_outside_training_range"].items())
        + ".",
    ]
    (res_dir / "report.md").write_text("\n".join(lines) + "\n")

    p = ["# Performance notes", "", "## Storage chunking", "", "| Variable | Chunk shape | Chunk MB | Chunks | Array MB |", "|---|---|---|---|---|"]
    for v, c in d["chunks"].items():
        p.append(f"| {v} | {tuple(c['chunk_shape'])} | {c['chunk_mb']:.2f} | {c['n_chunks']} | {c['array_mb']:.0f} |")
    p += [
        "",
        "## Dask chunk size vs graph size, time and memory (annual GDD on the main cube, each in a fresh process, local threaded scheduler)",
        "",
        "| Dask chunking | Chunk MB | Chunks per variable | Tasks (raw) | Tasks (optimised) | Wall s | Peak RSS increase MB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in perf["chunk_experiment"]:
        p.append(
            f"| {r['name']} | {r['chunk_mb']:.2f} | {r['n_chunks']} | {r['tasks_raw']:,} | {r['tasks_optimized']:,} | {r['wall_s']:.2f} | {r['peak_rss_increase_mb']:.0f} |"
        )
    if "out_of_core" in perf:
        s = perf["out_of_core"]["summary"]
        c = perf["out_of_core"]["compute"]
        p += [
            "",
            "## Out-of-core demonstration",
            "",
            f"- Cube: {s['cube_gb_uncompressed']:.2f} GB uncompressed ({s['disk_gb']:.2f} GB on disk), chunks {tuple(c['chunk']['chunk_shape'])} = {c['chunk']['chunk_mb']:.2f} MB.",
            f"- Every stage ran in its own process with a hard address-space cap of {s['cap_gb']:.2f} GB (RLIMIT_AS): the cube is {s['cube_to_cap_ratio']:.1f}x the cap.",
            f"- Writing the cube tile by tile: {s['write_wall_s']:.0f} s, peak RSS {s['write_peak_rss_mb']:.0f} MB.",
            f"- Naive `ds.load()`: **{s['naive_load_status']}**.",
            f"- Chunked GDD + heat days + JJA precip + SPI + annual precip: **{s['chunked_compute_status']}** in {s['compute_wall_s']:.1f} s, "
            f"peak RSS {s['chunked_peak_rss_mb']:.0f} MB ({100 * s['chunked_peak_rss_fraction_of_cube']:.1f}% of the cube).",
            f"- Graph tasks per feature: {c['graph_tasks_per_feature']}.",
        ]
        rd = perf["resample_vs_dot_big_cube"]
        p += [
            "",
            "## Seasonal sums on the big cube: `resample().sum()` vs indicator-matrix contraction (uncapped child processes)",
            "",
            "| Method | Tasks (raw) | Wall s | Peak RSS increase MB |",
            "|---|---|---|---|",
            *[f"| {k} | {rd[k]['tasks_raw']:,} | {rd[k]['wall_s']:.2f} | {rd[k]['peak_rss_increase_mb']:.0f} |" for k in ("resample", "dot")],
            "",
            f"Same result (checksums agree to 1e-6): {rd['same_result']}.",
        ]
    (res_dir / "performance.md").write_text("\n".join(p) + "\n")
