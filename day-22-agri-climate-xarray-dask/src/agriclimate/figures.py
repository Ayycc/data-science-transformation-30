"""Static figures (matplotlib, Agg) and a dependency-free interactive SVG/HTML exposure map."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, GRAY_MID = "#e34948", "#f0efec"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", [RED, GRAY_MID, BLUE])
SEQ_BLUE = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "font.size": 10,
    }
)


def _outline(ax, fields, **kw) -> None:
    for f in fields:
        g = f["geometry"]
        polys = getattr(g, "geoms", [g])
        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, **kw)


def drought_map(feats, nd, fields, year: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    spi = feats["spi3_aug"].sel(year=year)
    m = spi.plot.imshow(ax=axes[0], cmap=DIVERGING, norm=TwoSlopeNorm(0, -2.5, 2.5), add_colorbar=False)
    fig.colorbar(m, ax=axes[0], label="SPI-3 (Jun-Aug), SD units")
    anom = nd["ndvi_int"].sel(year=year) - nd["ndvi_int"].mean("year")
    m2 = anom.plot.imshow(ax=axes[1], cmap=DIVERGING, norm=TwoSlopeNorm(0, -25, 25), add_colorbar=False)
    fig.colorbar(m2, ax=axes[1], label="NDVI integral anomaly (NDVI-days)")
    for ax, t in zip(axes, (f"SPI-3 drought index, {year}", f"Seasonal NDVI integral anomaly, {year}")):
        _outline(ax, fields, color=INK, lw=0.5)
        ax.set_title("")
        ax.set_title(t + " (SYNTHETIC)", loc="left")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.grid(False)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cv_comparison(cv: dict, path: Path) -> None:
    sets = [("nowcast_with_ndvi", "Nowcast model\n(climate + NDVI)"), ("climate_only", "Climate-only model\n(used for scenarios)")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    x = np.arange(len(sets))
    w = 0.36
    for k, (scheme, color, label) in enumerate((("random", BLUE, "Random K-fold"), ("spatial", ORANGE, "Spatial block K-fold"))):
        vals = [cv[f"{s}__{scheme}"]["rmse"] for s, _ in sets]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(cv["baseline_mean"]["rmse"], color=INK2, lw=1, ls="--")
    ax.text(x[-1] + 0.45, cv["baseline_mean"]["rmse"], "predict-the-mean", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(x, [lbl for _, lbl in sets])
    ax.set_ylabel("out-of-fold RMSE (t/ha)")
    ax.set_title("Random CV understates error on unseen locations (SYNTHETIC yields)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def scenario_distribution(scen, risk_model: dict, risk_true: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    bins = np.linspace(min(scen["pred_scen"].min(), scen["true_scen"].min()) - 0.2, max(scen["pred_base"].max(), scen["true_base"].max()) + 0.2, 60)
    ax.hist(scen["pred_base"], bins=bins, color=BLUE, alpha=0.55, label="Baseline climate (model)")
    ax.hist(scen["pred_scen"], bins=bins, color=ORANGE, alpha=0.55, label="+2 C / -20% precip (model)")
    for r, color, name in ((risk_model["baseline"], BLUE, "baseline P5"), (risk_model["scenario"], ORANGE, "scenario P5")):
        ax.axvline(r["p5_t_ha"], color=color, lw=2)
        ax.text(r["p5_t_ha"], ax.get_ylim()[1] * 0.95, f" {name}\n {r['p5_t_ha']:.2f}", color=INK, fontsize=8, va="top")
    ax.axvline(risk_true["scenario"]["p5_t_ha"], color=INK2, lw=1, ls="--")
    ax.text(
        risk_true["scenario"]["p5_t_ha"],
        ax.get_ylim()[1] * 0.6,
        f" true scenario P5\n {risk_true['scenario']['p5_t_ha']:.2f}",
        color=INK2,
        fontsize=8,
    )
    ax.set_xlabel("yield, t/ha (field x weather-year, 2024 technology)")
    ax.set_ylabel("count")
    ax.set_title("Portfolio yield distribution under the climate scenario (SYNTHETIC)", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def chunking(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    names = [r["name"] for r in rows]
    for ax, key, title in zip(
        axes,
        ("wall_s", "tasks_raw", "peak_rss_increase_mb"),
        ("Wall time (s)", "Dask tasks in graph", "Peak RSS increase (MB)"),
    ):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=BLUE, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}" if key != "tasks_raw" else f"{v:,}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(rows)), [n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Annual GDD on the main cube under three Dask chunkings", x=0.01, ha="left")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def memory_profile(traces: dict, cap_gb: float, summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    offset = 0.0
    styles = {"write": (AQUA, "write cube tile by tile"), "load": (RED, "naive full load -> MemoryError"), "compute": (BLUE, "chunked indices")}
    for stage in ("write", "load", "compute"):
        tr = np.array(traces[stage]) if traces[stage] else np.zeros((1, 2))
        color, label = styles[stage]
        ax.plot(tr[:, 0] + offset, tr[:, 1], color=color, lw=2, label=f"{stage}: {label}")
        offset += tr[-1, 0] + 2
    ax.axhline(cap_gb * 1024, color=INK, lw=1, ls="--")
    ax.text(0, cap_gb * 1024, f" address-space cap {cap_gb:.2f} GB", va="bottom", fontsize=8)
    ax.axhline(summary["cube_gb_uncompressed"] * 1024, color=INK2, lw=1, ls=":")
    ax.text(
        0,
        summary["cube_gb_uncompressed"] * 1024,
        f" cube size {summary['cube_gb_uncompressed']:.2f} GB (uncompressed)",
        va="bottom",
        fontsize=8,
        color=INK2,
    )
    ax.set_ylim(0, summary["cube_gb_uncompressed"] * 1024 * 1.12)
    ax.set_xlabel("seconds (three separate capped processes, laid end to end)")
    ax.set_ylabel("resident memory (MB)")
    ax.set_title(f"Out-of-core run: cube is {summary['cube_to_cap_ratio']:.1f}x the memory cap", loc="left")
    ax.legend(frameon=False, loc="center right")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def exposure_map_png(per_field, fields, path: Path) -> None:
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    geoms = {f["field_id"]: f["geometry"] for f in fields}
    polys, vals = [], []
    for fid, row in per_field.iterrows():
        g = geoms[fid]
        for p in getattr(g, "geoms", [g]):
            polys.append(np.array(p.exterior.coords))
            vals.append(row["change_pct"])
    vmax = max(abs(min(vals)), 1.0)
    pc = PolyCollection(polys, array=np.array(vals), cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), edgecolor=SURFACE, linewidth=0.4)
    ax.add_collection(pc)
    ax.autoscale_view()
    fig.colorbar(pc, ax=ax, label="mean yield change under +2 C / -20% precip (%)")
    ax.set_title("Field-level climate exposure (SYNTHETIC fields)", loc="left")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _hex(v: float, vmax: float) -> str:
    rgba = DIVERGING(TwoSlopeNorm(0, -vmax, vmax)(v))
    r, g, b = (int(255 * c) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def exposure_map_html(per_field, fields, path: Path) -> None:
    """Self-contained interactive map: hover a field for its scenario numbers. No JS libraries, no tiles."""
    from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

    W, H = 900, 600
    sx = lambda lon: (lon - LON_MIN) / (LON_MAX - LON_MIN) * W  # noqa: E731
    sy = lambda lat: (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * H  # noqa: E731
    geoms = {f["field_id"]: f["geometry"] for f in fields}
    vmax = max(abs(per_field["change_pct"].min()), 1.0)
    shapes = []
    for fid, r in per_field.iterrows():
        g = geoms[fid]
        tip = html.escape(
            f"Field {fid} | baseline mean {r['base_mean']:.2f} t/ha | scenario mean {r['scen_mean']:.2f} t/ha "
            f"({r['change_pct']:+.1f}%) | scenario P5 {r['scen_p5']:.2f} t/ha | YaR95 {r['yar95_scen']:.2f} t/ha"
        )
        for p in getattr(g, "geoms", [g]):
            pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p.exterior.coords)
            shapes.append(f'<polygon points="{pts}" fill="{_hex(r["change_pct"], vmax)}" data-tip="{tip}"><title>{tip}</title></polygon>')
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Drought and heat exposure by field (SYNTHETIC)</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f9f9f7;color:#0b0b0b;margin:24px}}
svg{{background:#fcfcfb;border:1px solid #e4e3df}} polygon{{stroke:#fcfcfb;stroke-width:1}}
polygon:hover{{stroke:#0b0b0b;stroke-width:2}} #tip{{min-height:1.4em;color:#52514e;margin:8px 0}}
.legend span{{display:inline-block;width:28px;height:12px;margin-right:2px}}
</style></head><body>
<h2>Mean yield change per field under +2 &deg;C / &minus;20% precipitation (SYNTHETIC data)</h2>
<p>Each polygon is a synthetic field; colour is the model's mean yield change across 24 weather-years. Hover a field for details.</p>
<div id="tip">Hover a field</div>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="field exposure map">{"".join(shapes)}</svg>
<p class="legend"><span style="background:{_hex(-vmax, vmax)}"></span>{-vmax:.0f}% &nbsp;
<span style="background:{_hex(0, vmax)}"></span>0% &nbsp; <span style="background:{_hex(vmax, vmax)}"></span>+{vmax:.0f}%
&nbsp; | lon {LON_MIN}..{LON_MAX}, lat {LAT_MIN}..{LAT_MAX}</p>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('polygon').forEach(p=>p.addEventListener('mouseenter',()=>tip.textContent=p.dataset.tip));
</script></body></html>"""
    Path(path).write_text(doc)
