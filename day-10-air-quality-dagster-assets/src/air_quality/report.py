"""Publishes the daily AQI report, alert list and figures from the DuckDB warehouse."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from .config import END_DATE_EXCLUSIVE, START_DATE  # noqa: E402
from .resources import ReportStore  # noqa: E402

BLOCKING_CHECKS = ("raw_schema", "raw_completeness", "value_ranges", "systemic_fault_rate", "aqi_coverage")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _tables(conn) -> set[str]:
    return set(conn.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])


def daily_summary(conn) -> pd.DataFrame:
    t = _tables(conn)
    dates = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE_EXCLUSIVE, inclusive="left").date})
    out = dates.copy()
    if "check_results" in t:
        chk = conn.execute("SELECT date, check_name, passed, value FROM check_results").df()
        chk["date"] = pd.to_datetime(chk["date"]).dt.date
        val = chk.pivot_table(index="date", columns="check_name", values="value", aggfunc="last")
        keep = [c for c in ("raw_completeness", "systemic_fault_rate", "aqi_coverage") if c in val.columns]
        out = out.merge(val[keep].reset_index(), on="date", how="left")
        failed = chk[(~chk["passed"]) & chk["check_name"].isin(BLOCKING_CHECKS)]
        blocked = (
            failed.groupby("date")["check_name"].apply(lambda s: ",".join(sorted(s))).rename("failed_blocking_checks")
        )
        out = out.merge(blocked.reset_index(), on="date", how="left")
    if "daily_aqi" in t:
        a = conn.execute(
            "SELECT date, count(aqi) AS stations_with_aqi, median(aqi) AS median_aqi, max(aqi) AS max_aqi FROM daily_aqi GROUP BY date"
        ).df()
        a["date"] = pd.to_datetime(a["date"]).dt.date
        out = out.merge(a, on="date", how="left")
    if "alert_publications" in t:
        p = conn.execute("SELECT date, station_alerts FROM alert_publications").df()
        p["date"] = pd.to_datetime(p["date"]).dt.date
        p["alerts_published"] = True
        out = out.merge(p, on="date", how="left")
        out["alerts_published"] = out["alerts_published"].astype("boolean").fillna(False).astype(bool)
    else:
        out["alerts_published"] = False
        out["station_alerts"] = np.nan
    if "region_advisories" in t:
        r = conn.execute("SELECT date, count(*) AS regional_advisories FROM region_advisories GROUP BY date").df()
        r["date"] = pd.to_datetime(r["date"]).dt.date
        out = out.merge(r, on="date", how="left")
        out.loc[out["alerts_published"], "regional_advisories"] = out.loc[
            out["alerts_published"], "regional_advisories"
        ].fillna(0)
    if "failed_blocking_checks" not in out:
        out["failed_blocking_checks"] = None
    return out


def _figures(conn, summary: pd.DataFrame) -> dict[str, bytes]:
    figs: dict[str, bytes] = {}
    t = _tables(conn)
    if "daily_aqi" not in t:
        return figs
    reg = conn.execute(
        "SELECT a.date, s.region, median(a.aqi) AS median_aqi FROM daily_aqi a JOIN stations s USING (station_id) GROUP BY 1, 2 ORDER BY 1, 2"
    ).df()
    reg["date"] = pd.to_datetime(reg["date"])
    piv = reg.pivot(index="region", columns="date", values="median_aqi").reindex(
        columns=pd.date_range(START_DATE, END_DATE_EXCLUSIVE, inclusive="left")
    )
    labels = [d.strftime("%m-%d") for d in piv.columns]

    heat = go.Figure(
        go.Heatmap(
            z=piv.values,
            x=labels,
            y=piv.index.tolist(),
            colorscale="YlOrRd",
            zmin=0,
            zmax=200,
            colorbar=dict(title="median AQI"),
        )
    )
    heat.update_layout(
        title="SYNTHETIC data: median daily AQI by region (blank = partition blocked by an asset check)", height=420
    )
    figs["figures/aqi_heatmap.html"] = heat.to_html(include_plotlyjs="cdn", full_html=True).encode()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    im = ax.imshow(np.ma.masked_invalid(piv.values), aspect="auto", cmap="YlOrRd", vmin=0, vmax=200)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xticks(range(0, len(labels), 2), labels[::2], rotation=45, fontsize=8)
    ax.set_title("SYNTHETIC data: median daily AQI by region (grey = blocked partition)")
    ax.set_facecolor("#d9d9d9")
    fig.colorbar(im, ax=ax, label="median AQI")
    ax = axes[1]
    s = summary.copy()
    x = np.arange(len(s))
    published = s["alerts_published"].to_numpy()
    ax.bar(
        x[published], s.loc[published, "station_alerts"].fillna(0), color="#d95f02", label="station alerts published"
    )
    ymax = max(1.0, float(s["station_alerts"].fillna(0).max()))
    ax.bar(
        x[~published],
        np.full((~published).sum(), ymax * 0.15),
        color="#7f7f7f",
        hatch="//",
        label="blocked by asset check",
    )
    ax.set_xticks(x[::2], [d.strftime("%m-%d") for d in s["date"]][::2], rotation=45, fontsize=8)
    ax.set_ylabel("station alerts (AQI > 100)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    figs["figures/aqi_overview.png"] = buf.getvalue()
    return figs


def publish(conn, store: ReportStore) -> dict[str, str]:
    """Write daily_summary.csv, alerts.csv, region_advisories.csv, aqi_report.md and figures."""
    written: dict[str, str] = {}
    summary = daily_summary(conn)
    written["daily_summary"] = store.write_csv("daily_summary.csv", summary)
    t = _tables(conn)
    alerts = (
        conn.execute("SELECT * FROM alerts ORDER BY date, aqi DESC, station_id").df()
        if "alerts" in t
        else pd.DataFrame()
    )
    adv = (
        conn.execute("SELECT * FROM region_advisories ORDER BY date, region").df()
        if "region_advisories" in t
        else pd.DataFrame()
    )
    written["alerts"] = store.write_csv("alerts.csv", alerts)
    written["region_advisories"] = store.write_csv("region_advisories.csv", adv)
    for name, data in _figures(conn, summary).items():
        written[name] = store.write_bytes(name, data)

    pub = summary[summary["alerts_published"]]
    blocked = summary[~summary["alerts_published"]]
    lines = [
        "# Daily AQI report (SYNTHETIC data)",
        "",
        f"Partitions {START_DATE} to {summary['date'].max()}: {len(pub)} published, {len(blocked)} blocked by a failed blocking asset check.",
        "",
    ]
    if len(pub):
        last = pub["date"].max()
        la = alerts[pd.to_datetime(alerts["date"]).dt.date == last] if len(alerts) else alerts
        lines += [f"## Latest published day: {last}", "", f"{len(la)} station alert(s) with AQI > 100.", ""]
        if len(la):
            lines += [md_table(la[["station_id", "region", "aqi", "category", "dominant_pollutant"]].head(20)), ""]
        with_alerts = pub[pub["station_alerts"].fillna(0) > 0]
        if len(with_alerts) and with_alerts["date"].max() != last:
            d = with_alerts["date"].max()
            la = alerts[pd.to_datetime(alerts["date"]).dt.date == d]
            lines += [f"## Most recent day with alerts: {d}", "", f"{len(la)} station alert(s); top 15 by AQI:", ""]
            lines += [md_table(la[["station_id", "region", "aqi", "category", "dominant_pollutant"]].head(15)), ""]
    if len(blocked):
        lines += [
            "## Blocked partitions (alerts withheld)",
            "",
            md_table(blocked[["date", "failed_blocking_checks"]].fillna("not materialised")),
            "",
        ]
    cols = [
        c
        for c in [
            "date",
            "raw_completeness",
            "systemic_fault_rate",
            "aqi_coverage",
            "stations_with_aqi",
            "median_aqi",
            "max_aqi",
            "station_alerts",
            "regional_advisories",
            "alerts_published",
        ]
        if c in summary
    ]
    tbl = summary[cols].copy()
    for c in ("median_aqi", "max_aqi", "stations_with_aqi", "station_alerts", "regional_advisories"):
        if c in tbl:
            tbl[c] = tbl[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    lines += ["## Day by day", "", md_table(tbl), ""]
    written["aqi_report"] = store.write_text("aqi_report.md", "\n".join(lines))
    return written
