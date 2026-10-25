"""Interactive folium maps and static matplotlib figures.

Maps embed every marker, hexagon and route as vector data; the basemap is an optional tile layer that
the viewer's browser fetches. With no network the tiles are blank but all data layers still render.
"""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import routing  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#f0efec"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("drain_fill", [RED, GRAY, BLUE])
POLICY_COLORS = {"none": "#8a8984", "fixed_routes": ORANGE, "dynamic_ha": YELLOW, "dynamic_lgbm": BLUE, "dynamic_oracle": AQUA}
TRUCK_COLORS = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", RED]


def _hex(v: float, vmax: float) -> str:
    rgba = DIVERGING(TwoSlopeNorm(0, -vmax, vmax)(np.clip(v, -vmax, vmax)))
    return matplotlib.colors.to_hex(rgba)


def _base_map(stations: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=[stations["lat"].median(), stations["lon"].median()], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap basemap (needs network)", control=True).add_to(m)
    return m


def feature_map(stations: pd.DataFrame, cells: gpd.GeoDataFrame, am_net: pd.Series, path: Path) -> None:
    """H3 cells coloured by mean weekday 07-10h net flow per station; stations with their features."""
    m = _base_map(stations)
    vmax = float(np.nanpercentile(np.abs(cells["am_net_per_station"]), 95)) or 1.0
    fg = folium.FeatureGroup(name="H3 res-8 cells: weekday 07-10h net flow per dock (bikes/h)")
    for _, c in cells.iterrows():
        folium.GeoJson(
            c.geometry.__geo_interface__,
            style_function=lambda _, col=_hex(c["am_net_per_station"], vmax): {
                "fillColor": col,
                "color": "#ffffff",
                "weight": 1,
                "fillOpacity": 0.75,
            },
            tooltip=f"{c['h3']}<br>stations: {c['n_stations']}<br>AM net/station: {c['am_net_per_station']:+.2f} bikes/h<br>"
            f"mean elevation: {c['elevation_m']:.0f} m<br>nearest Metro dock: {c['metro_dist_m']:.0f} m",
        ).add_to(fg)
    fg.add_to(m)
    sg = folium.FeatureGroup(name="Stations")
    for i, s in stations.iterrows():
        folium.CircleMarker(
            [s["lat"], s["lon"]],
            radius=3 + 0.15 * s["capacity"],
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=_hex(am_net.iat[i], vmax * 2),
            fill_opacity=0.95,
            tooltip=f"{s['name']}<br>docks (est.): {s['capacity']}<br>AM net: {am_net.iat[i]:+.2f} bikes/h<br>"
            f"elevation {s['elevation_m']:.0f} m (rel. to ring: {s['elev_rel_ring2']:+.0f} m)<br>Metro: {s['metro_dist_m']:.0f} m",
        ).add_to(sg)
    sg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))


def routes_map(stations: pd.DataFrame, plan: routing.RoutePlan, depot: tuple[float, float], title: str, path: Path) -> None:
    m = _base_map(stations)
    folium.Marker(depot, tooltip="Depot", icon=folium.Icon(color="black", icon="home")).add_to(m)
    for v, stops in enumerate(plan.routes):
        col = TRUCK_COLORS[v % len(TRUCK_COLORS)]
        fg = folium.FeatureGroup(name=f"Truck {v + 1}: {len(stops)} stops, {sum(abs(s.move) for s in stops)} bikes")
        pts = [depot] + [(stations.at[s.station, "lat"], stations.at[s.station, "lon"]) for s in stops] + [depot]
        folium.PolyLine(pts, color=col, weight=3, opacity=0.85).add_to(fg)
        for k, s in enumerate(stops, 1):
            lat, lon = stations.at[s.station, "lat"], stations.at[s.station, "lon"]
            verb = f"deliver {s.move}" if s.move > 0 else f"pick up {-s.move}"
            folium.CircleMarker(
                [lat, lon],
                radius=4 + abs(s.move) / 2,
                color=col,
                weight=2,
                fill=True,
                fill_color=BLUE if s.move > 0 else RED,
                fill_opacity=0.9,
                tooltip=f"Truck {v + 1} stop {k} (+{s.eta_s // 60} min): {verb}<br>{stations.at[s.station, 'name']}",
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(
        folium.Element(
            f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:#fff;padding:6px 10px;font:14px sans-serif'>{title}</div>"
        )
    )
    m.save(str(path))


def before_after_map(stations: pd.DataFrame, results: dict, path: Path) -> None:
    """One layer per policy: circle area ~ stockout hours (empty + full) at each station."""
    m = _base_map(stations)
    for k, (name, r) in enumerate(results.items()):
        hours = (r.empty_min + r.full_min) / 60
        fg = folium.FeatureGroup(name=f"{name}: {hours.sum():,.0f} stockout h", show=k in (0, len(results) - 1))
        for i, s in stations.iterrows():
            if hours[i] < 0.5:
                continue
            folium.CircleMarker(
                [s["lat"], s["lon"]],
                radius=float(np.sqrt(hours[i])) * 0.9,
                color=POLICY_COLORS.get(name, BLUE),
                weight=1,
                fill=True,
                fill_opacity=0.45,
                tooltip=f"{name}<br>{s['name']}<br>empty {r.empty_min[i] / 60:.1f} h, full {r.full_min[i] / 60:.1f} h, lost trips {r.lost[i]}",
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))


# --------------------------------------------------------------------------------------------- static figures
def _style(ax: plt.Axes) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_forecast(times: pd.DatetimeIndex, actual: np.ndarray, preds: dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=130)
    ax.bar(times, actual, width=1 / 24 * 0.8, color="#c3c2b7", label="actual net flow")
    for (name, p), col in zip(preds.items(), (BLUE, ORANGE), strict=False):
        ax.plot(times, p, color=col, linewidth=2, label=name)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("bikes / hour (arrivals - departures)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_bars(labels: list[str], values: list[float], colors: list[str], ylabel: str, title: str, path: Path, fmt: str = "{:.3f}") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=130)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_stockouts(summary: dict[str, dict], path: Path) -> None:
    names = list(summary)
    empty = [summary[n]["empty_dock_hours"] for n in names]
    full = [summary[n]["full_dock_hours"] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=130)
    x = np.arange(len(names))
    ax.bar(x, empty, width=0.6, color=RED, label="empty-dock hours")
    ax.bar(x, full, width=0.6, bottom=np.array(empty) + 0, color=BLUE, label="full-dock hours", edgecolor="#ffffff", linewidth=1.5)
    for xi, e, f in zip(x, empty, full, strict=True):
        ax.text(xi, e + f, f"{e + f:,.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [n.replace("_", "\n", 1) for n in names])
    ax.set_ylabel("station-hours, 14-day test", color=INK2, fontsize=9)
    ax.set_title("Simulated stockout hours by rebalancing policy", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_importance(imp: pd.Series, path: Path, top: int = 15) -> None:
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.barh(s.index, s.to_numpy(), color=BLUE, height=0.6)
    ax.set_xlabel("share of total gain", color=INK2, fontsize=9)
    ax.set_title("LightGBM feature importance (gain)", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_routes_static(
    stations: pd.DataFrame, cells: gpd.GeoDataFrame, plan: routing.RoutePlan, depot: tuple[float, float], title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    vmax = float(np.nanpercentile(np.abs(cells["am_net_per_station"]), 95)) or 1.0
    cells.plot(
        ax=ax,
        column="am_net_per_station",
        cmap=DIVERGING,
        vmin=-vmax,
        vmax=vmax,
        edgecolor="#ffffff",
        linewidth=0.4,
        legend=True,
        legend_kwds={"label": "weekday 07-10h net flow per dock (bikes/h)", "shrink": 0.6},
    )
    ax.scatter(stations["lon"], stations["lat"], s=2, color=INK2)
    for v, stops in enumerate(plan.routes):
        pts = np.array([depot] + [(stations.at[s.station, "lat"], stations.at[s.station, "lon"]) for s in stops] + [depot])
        ax.plot(pts[:, 1], pts[:, 0], color=TRUCK_COLORS[v % 8], linewidth=1.6)
    ax.scatter([depot[1]], [depot[0]], marker="s", s=60, color=INK, zorder=5)
    bounds = stations[["lon", "lat"]].quantile([0.01, 0.99])
    ax.set_xlim(bounds["lon"].iat[0] - 0.01, bounds["lon"].iat[1] + 0.01)
    ax.set_ylim(bounds["lat"].iat[0] - 0.01, bounds["lat"].iat[1] + 0.01)
    ax.set_aspect(1 / np.cos(np.radians(38.9)))
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
