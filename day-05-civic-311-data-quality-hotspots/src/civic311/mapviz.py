"""Offline hotspot map: one self-contained HTML file (inline SVG + CSS, no scripts, tiles or CDNs)."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from . import geo

W, H, PAD = 760, 900, 20


def _projector():
    kx = geo.km_per_deg_lon()
    ky = geo.KM_PER_DEG_LAT
    span_x = (geo.LON_MAX - geo.LON_MIN) * kx
    span_y = (geo.LAT_MAX - geo.LAT_MIN) * ky
    s = min((W - 2 * PAD) / span_x, (H - 2 * PAD) / span_y)

    def p(lon: float, lat: float) -> tuple[float, float]:
        return PAD + (lon - geo.LON_MIN) * kx * s, H - PAD - (lat - geo.LAT_MIN) * ky * s

    return p, s


def _hex(r: float, g: float, b: float) -> str:
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _color(z: float, cls: str) -> str:
    if cls == "hot":
        t = min(abs(z) / 15, 1)
        return _hex(214 - 60 * t, 96 - 70 * t, 77 - 50 * t)
    if cls == "cold":
        t = min(abs(z) / 15, 1)
        return _hex(84 - 50 * t, 140 - 60 * t, 200 - 40 * t)
    if cls == "significant_not_material":
        return "#e9c46a" if z > 0 else "#a8c5da"
    return "#d9d9d9"


def hotspot_map_html(cells: pd.DataFrame, path: Path, title: str, subtitle: str) -> None:
    proj, scale = _projector()
    polys = []
    for r in cells.itertuples():
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (proj(lo, la) for lo, la in geo.cell_boundary_lonlat(r.h3)))
        tip = (
            f"{r.h3}\n{r.gi_class.replace('_', ' ')}\nrequests: {r.n:,}\nmedian hours: {r.median_hours:.1f}\n"
            f"cell excess vs city: {r.cell_excess_pct:+.0f}%\nGi* window excess: {r.local_excess_pct:+.0f}%\n"
            f"Gi* z: {r.gi_z:.1f}  q: {r.gi_q:.2g}"
        )
        polys.append(f'<polygon class="c" points="{pts}" fill="{_color(r.gi_z, r.gi_class)}">'
                     f"<title>{html.escape(tip)}</title></polygon>")  # fmt: skip
    city = " ".join(f"{x:.1f},{y:.1f}" for x, y in (proj(lo, la) for lo, la in geo.CITY_LONLAT))
    zones = []
    for name, lon, lat, r_km, mult in geo.PLANTED_ZONES:
        cx, cy = proj(lon, lat)
        zones.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_km * scale:.1f}" class="z"><title>planted (synthetic truth): '
            f"{name}, x{mult}</title></circle>"
            f'<text x="{cx + r_km * scale + 3:.1f}" y="{cy:.1f}" class="zl">{html.escape(name)} x{mult}</text>'
        )
    counts = cells.gi_class.value_counts()
    legend = [
        ("hot", "#b43c2a", "hot spot: Gi* q<0.05 and window >= 25% slower"),
        ("cold", "#3a6fb0", "cold spot: Gi* q<0.05 and window >= 20% faster"),
        ("significant_not_material", "#e9c46a", "significant, below materiality"),
        ("not_significant", "#d9d9d9", "not significant"),
    ]
    leg = "".join(
        f'<li><span style="background:{c}"></span>{html.escape(t)} ({int(counts.get(k, 0))} cells)</li>'
        for k, c, t in legend
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1f2328;--muted:#5b616b;--line:#9aa0a6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e6e3;--muted:#a3a7ad;--line:#6b7078}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:16px;display:flex;flex-wrap:wrap;gap:16px}}
svg{{width:100%;max-width:{W}px;height:auto;flex:1 1 420px}}
aside{{flex:1 1 260px;min-width:0}}
h1{{font-size:20px;margin:16px 16px 0}} p{{color:var(--muted)}}
.c{{stroke:var(--bg);stroke-width:.6}} .c:hover{{stroke:var(--fg);stroke-width:1.6}}
.city{{fill:none;stroke:var(--line);stroke-width:1.2}}
.z{{fill:none;stroke:var(--fg);stroke-dasharray:4 3;stroke-width:1.2}} .zl{{fill:var(--fg);font-size:11px}}
ul{{list-style:none;padding:0}} li{{margin:6px 0;display:flex;gap:8px;align-items:center}}
li span{{display:inline-block;width:14px;height:14px;border-radius:2px;flex:none}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<main>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="H3 hexagon map of 311 resolution-time hot spots">
<polygon class="city" points="{city}"/>
{"".join(polys)}
{"".join(zones)}
</svg>
<aside>
<p>{html.escape(subtitle)}</p>
<ul>{leg}</ul>
<p>Dashed circles: zones planted by the SYNTHETIC generator (ground truth, not visible to the analysis).
Grey outline: simplified city boundary. Hover a hexagon for its numbers. This file has no external
dependencies and renders offline.</p>
</aside>
</main></body></html>
"""
    path.write_text(doc)


def hotspot_map_png(cells: pd.DataFrame, path: Path, title: str) -> None:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Point, Polygon

    gdf = gpd.GeoDataFrame(
        cells.assign(color=[_color(z, c) for z, c in zip(cells.gi_z, cells.gi_class, strict=True)]),
        geometry=[Polygon(geo.cell_boundary_lonlat(c)) for c in cells.h3],
        crs="EPSG:4326",
    ).to_crs("EPSG:26971")
    city = gpd.GeoSeries([geo.CITY], crs="EPSG:4326").to_crs("EPSG:26971")
    zones = gpd.GeoSeries([Point(lo, la) for _, lo, la, _, _ in geo.PLANTED_ZONES], crs="EPSG:4326").to_crs(
        "EPSG:26971"
    )
    fig, ax = plt.subplots(figsize=(7, 8.5))
    gdf.plot(ax=ax, color=gdf.color, edgecolor="white", linewidth=0.3)
    city.boundary.plot(ax=ax, color="#555", linewidth=0.8)
    for (name, _, _, r_km, m), pt in zip(geo.PLANTED_ZONES, zones, strict=True):
        ax.add_patch(plt.Circle((pt.x, pt.y), r_km * 1000, fill=False, ls="--", lw=1, color="black"))
        ax.annotate(f"{name} x{m}", (pt.x + r_km * 1000, pt.y), fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_geojson(cells: pd.DataFrame, path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import Polygon

    cols = ["h3", "n", "median_hours", "cell_excess_pct", "local_excess_pct", "gi_z", "gi_q", "gi_class"]
    gdf = gpd.GeoDataFrame(
        cells[cols].round(4), geometry=[Polygon(geo.cell_boundary_lonlat(c)) for c in cells.h3], crs="EPSG:4326"
    )
    gdf["geometry"] = gdf.geometry.set_precision(1e-6)
    path.write_text(gdf.to_json(drop_id=True))
