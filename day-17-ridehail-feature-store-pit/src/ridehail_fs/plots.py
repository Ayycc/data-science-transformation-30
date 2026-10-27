"""Static figures (matplotlib, PNG). Two-series charts use a fixed categorical order:
slot 1 blue = point-in-time (correct), slot 2 orange = naive (leaky)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def offline_vs_production(res: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    _style(ax)
    groups = ["Offline validation\n(as the team measured it)", "Production\n(features as served)"]
    naive = [res["naive"]["offline"]["mae_min"], res["naive"]["production"]["mae_min"]]
    pit = [res["pit"]["offline"]["mae_min"], res["pit"]["production"]["mae_min"]]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, pit, w, color=BLUE, label="Point-in-time join (Feast)")
    b2 = ax.bar(x + w / 2 + 0.01, naive, w, color=ORANGE, label="Naive hour-bucket join")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylabel("MAE, minutes (test week)", color=INK2, fontsize=9)
    ax.set_title("ETA error: the naive model's offline score does not survive production", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, max(naive + pit) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def latency_cdf(lat_ms: np.ndarray, p50: float, p99: float, out: Path, budget_ms: float = 20.0) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style(ax)
    s = np.sort(lat_ms)
    ax.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=BLUE, linewidth=2)
    ax.axvline(budget_ms, color=GRAY, linewidth=1, linestyle="--")
    ax.text(budget_ms, 50, f" {budget_ms:.0f} ms budget", color=INK2, fontsize=9, va="center")
    ax.annotate(f"p50 {p50:.2f} ms", (p50, 50), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
    ax.annotate(f"p99 {p99:.2f} ms", (p99, 99), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("get_online_features latency, ms (log scale, single request, SQLite online store)", color=INK2, fontsize=9)
    ax.set_ylabel("% of requests", color=INK2, fontsize=9)
    ax.set_title("Online retrieval latency (feature service incl. on-demand transform)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def leakage_timeline(zone_traffic: pd.DataFrame, hourly: pd.DataFrame, trips: pd.DataFrame, zone_id: int, day: pd.Timestamp, out: Path) -> None:
    """What each join 'knew' about traffic in one zone over one day vs the speed trips actually achieved."""
    lo, hi = day, day + pd.Timedelta(days=1)
    zt = zone_traffic[(zone_traffic.zone_id == zone_id) & zone_traffic.event_timestamp.between(lo, hi)]
    hz = hourly[(hourly.zone_id == zone_id) & hourly.hour.between(lo, hi)]
    tr = trips[(trips.zone_id == zone_id) & trips.request_ts.between(lo, hi)]
    fig, ax = plt.subplots(figsize=(8.4, 3.8), facecolor=SURFACE)
    _style(ax)

    def nv(x: pd.Series) -> pd.Series:
        return pd.to_datetime(x, utc=True).dt.tz_convert(None)

    ax.scatter(nv(tr.request_ts), tr.distance_km / (tr.duration_min / 60.0), s=10, color=GRAY, alpha=0.6, label="Trips starting here: realised speed")
    ax.step(nv(zt.event_timestamp), zt.avg_speed_15m, where="post", color=BLUE, linewidth=2, label="PIT: last 15-min speed known at request")
    if len(hz):
        h = nv(hz.hour)
        xs = pd.concat([h, pd.Series([h.max() + pd.Timedelta(hours=1)])], ignore_index=True)
        ax.step(
            xs,
            np.r_[hz.avg_speed_15m.to_numpy(), hz.avg_speed_15m.to_numpy()[-1]],
            where="post",
            color=ORANGE,
            linewidth=2,
            label="Naive: end-of-hour value joined on clock hour",
        )
    ax.set_ylabel("km/h", color=INK2, fontsize=9)
    ax.set_title(f"Zone {zone_id}, {day:%a %d %b}: the naive join reads traffic from later in the hour", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
