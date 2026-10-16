"""Figures, tables and the Markdown/HTML reports written to results/."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import contract, generate, mapviz  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def agreement(rejects: pd.DataFrame, ge: dict) -> dict:
    """pandera (enforcement) vs Great Expectations (audit) counts per rule on the raw batch."""
    pc = rejects.reason_code.value_counts()
    ge_df = pd.DataFrame(ge["results"])
    gsum = ge_df.groupby("rule_id").unexpected_count.sum()
    rows = []
    for r in contract.RULES:
        p, g = int(pc.get(r.code, 0)), int(gsum.get(r.rule_id, 0))
        rows.append({"rule_id": r.rule_id, "code": r.code, "pandera_rows": p, "ge_unexpected": g,
                     "comparable": r.ge_comparable, "match": (p == g) if r.ge_comparable else None})  # fmt: skip
    comp = [x for x in rows if x["comparable"]]
    return {"rows": rows, "summary": {"comparable_rules": len(comp), "matching": sum(bool(x["match"]) for x in comp)}}


def _fig_quality(daily: pd.DataFrame, q_anoms: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[1, 1.3])
    ax = axes[0]
    ax.plot(daily.index, 100 * daily.pass_rate, lw=1, color="#2a6f97")
    ax.set_ylabel("rows passing contract (%)")
    ax.set_title("Daily data-quality KPIs (SYNTHETIC extract; shaded = injected incidents, x = flagged anomaly)")
    for inc in generate.INCIDENTS:
        for a in axes:
            a.axvspan(pd.Timestamp(inc["start"]), pd.Timestamp(inc["end"]) + pd.Timedelta(days=1), color="#f4a261",
                      alpha=0.3, lw=0)  # fmt: skip
        ax.annotate(inc["id"], (pd.Timestamp(inc["start"]), ax.get_ylim()[0]), fontsize=8, va="bottom")
    ax = axes[1]
    for code, color in (("COORDS_MISSING", "#264653"), ("CLOSED_BEFORE_CREATED", "#e76f51"),
                        ("WARD_INVALID", "#2a9d8f")):  # fmt: skip
        s = 100 * daily[f"rate_{code}"]
        ax.plot(daily.index, s, lw=0.9, color=color, label=code)
        f = q_anoms[q_anoms.kpi == code]
        ax.scatter(f.date, 100 * f.rate, marker="x", color=color, s=30, zorder=3)
    ax.set_ylabel("rule failure rate (% of rows)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_reasons(reason_summary: pd.DataFrame, path: Path) -> None:
    s = reason_summary.sort_values("reason_rows")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(s.rule_id + " " + s.reason_code, s.reason_rows, color="#2a6f97")
    for y, v in enumerate(s.reason_rows):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8)
    ax.set_xlabel("quarantined rows citing the rule (a row can cite several)")
    ax.set_title("Quarantine table by rule (SYNTHETIC extract)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_volume(df: pd.DataFrame, vol: pd.DataFrame, path: Path) -> None:
    d = df.created_ts.dt.normalize().value_counts().sort_index()
    d = d[(d.index >= "2025-01-01") & (d.index < "2026-01-01")]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(d.index, d.values, lw=0.8, color="#555")
    f = vol[vol.series == "ALL"]
    ax.scatter(f.date, f["count"], color="#e63946", zorder=3, label=f"flagged (robust z >= {generate_vz()})")
    ax.set_ylabel("requests per day")
    ax.set_title("Operational KPI: daily request volume vs same-weekday baseline (SYNTHETIC; storm on 15-16 Jul)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_vz() -> float:
    from .config import VOLUME_Z

    return VOLUME_Z


def _fig_slo(slo: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    for ax, (q, g) in zip(axes, slo.groupby("quarter"), strict=False):
        ax.plot(g.date, 100 * g.ideal_left, ls="--", color="#999", lw=1, label="linear burn")
        ax.plot(g.date, 100 * g.budget_left, color="#e76f51" if g.budget_left.iloc[-1] < 0 else "#2a9d8f", lw=1.5,
                label="budget left")  # fmt: skip
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_title(f"{q}: {100 * g.budget_left.iloc[-1]:.0f}% left", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("error budget left (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Data-quality SLO burn-down: >= 90% of each quarter's rows pass the contract (SYNTHETIC)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_equity(eq: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (scope, g) in zip(axes, eq.groupby("scope", sort=False), strict=True):
        colors = g.income_quartile.map({"Q1": "#e76f51", "Q2": "#f4a261", "Q3": "#8ab17d", "Q4": "#2a9d8f"})
        ax.scatter(g.median_income / 1000, g.excess_pct, c=colors, s=18 + g.n / g.n.max() * 60)
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(scope.replace("_", " "), fontsize=9)
        ax.set_xlabel("SYNTHETIC median household income ($k)")
    axes[0].set_ylabel("category-adjusted delay vs city (%)")
    fig.suptitle("Equity: neighbourhood resolution delay vs income (colour = income quartile, size = requests)",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v).replace("|", "\\|")


def _validation_report(results: Path, metrics: dict, agree: dict, ge_raw: dict, ge_clean: dict,
                       reason_summary: pd.DataFrame) -> None:  # fmt: skip
    rules = pd.DataFrame([{"rule_id": r.rule_id, "code": r.code, "column": r.column, "check": r.description,
                           "GE expectation": r.ge} for r in contract.RULES])  # fmt: skip
    ag = pd.DataFrame(agree["rows"])
    ge_r = pd.DataFrame(ge_raw["results"])[["rule_id", "expectation", "success", "element_count", "unexpected_count"]]
    ge_c = pd.DataFrame(ge_clean["results"])[["rule_id", "expectation", "success", "element_count", "unexpected_count"]]
    q = metrics["quarantine"]
    md = f"""# Validation report: {contract.CONTRACT_VERSION}

SYNTHETIC extract, {metrics["data"]["rows_raw"]:,} rows ({metrics["data"]["scale"]} scale). Generated by `make run`.

- pandera (enforcement): {q["rows_quarantined"]:,} rows quarantined ({q["pct_quarantined"]}%), {q["reason_rows"]:,} reason rows, {q["rows_clean"]:,} clean rows.
- Great Expectations (audit), raw batch: {ge_raw["successful_expectations"]}/{ge_raw["evaluated_expectations"]} expectations pass.
- Great Expectations, clean batch: {ge_clean["successful_expectations"]}/{ge_clean["evaluated_expectations"]} expectations pass.
- pandera vs GE agreement on comparable rules: {agree["summary"]["matching"]}/{agree["summary"]["comparable_rules"]}.

## Contract rules

{_md_table(rules)}

## Quarantine by rule (pandera)

{_md_table(reason_summary)}

## pandera vs Great Expectations on the raw batch

Rules marked not comparable count something different by design: GE `expect_column_values_to_be_unique` flags every copy
of a duplicated id (pandera keeps the first copy), and rules implemented with two GE expectations (one per coordinate)
can count the same row twice.

{_md_table(ag)}

## Great Expectations: raw batch

{_md_table(ge_r)}

## Great Expectations: clean batch (after quarantine)

{_md_table(ge_c)}
"""  # noqa: E501
    (results / "validation_report.md").write_text(md)
    body = "".join(
        f"<h2>{html.escape(t)}</h2>{d.to_html(index=False, border=0, classes='t')}"
        for t, d in (("Contract rules", rules), ("Quarantine by rule (pandera)", reason_summary),
                     ("pandera vs Great Expectations (raw batch)", ag), ("Great Expectations: raw batch", ge_r),
                     ("Great Expectations: clean batch", ge_c))
    )  # fmt: skip
    head = md.split("## Contract rules")[0].replace("# ", "", 1).split("\n", 1)
    summary_items = "".join(
        f"<li>{html.escape(x[2:])}</li>" for x in head[1].strip().splitlines() if x.startswith("- ")
    )
    (results / "validation_report.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Validation report</title><style>
:root{{--bg:#fff;--fg:#1f2328;--line:#d0d7de}}@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e6e3;
--line:#3a3f45}}}}body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;max-width:1100px;
margin:0 auto;padding:16px}}table{{border-collapse:collapse;display:block;overflow-x:auto}}td,th{{border-bottom:1px
solid var(--line);padding:4px 8px;text-align:left;font-size:12px}}</style></head><body><h1>{html.escape(head[0])}</h1>
<p>SYNTHETIC data.</p><ul>{summary_items}</ul>{body}</body></html>"""
    )


def write_all(results: Path, *, metrics, rejects, reason_summary, examples, detection, mapping, reviewed, ge_raw,
              ge_clean, daily, q_anoms, inc_table, vol, slo, cells, eq_table, df) -> None:  # fmt: skip
    agree = agreement(rejects, ge_raw)
    reason_summary.to_csv(results / "quarantine_by_rule.csv", index=False)
    examples.to_csv(results / "quarantine_examples.csv", index=False)
    detection.to_csv(results / "defect_detection.csv", index=False)
    mapping.sort_values(["method", "raw_value"]).to_csv(results / "category_mapping_applied.csv", index=False)
    (results / "ge_validation_raw.json").write_text(json.dumps(ge_raw, indent=1) + "\n")
    (results / "ge_validation_clean.json").write_text(json.dumps(ge_clean, indent=1) + "\n")
    daily.round(6).to_csv(results / "kpi_daily.csv")
    q_anoms.round(5).to_csv(results / "kpi_quality_anomalies.csv", index=False)
    inc_table.to_csv(results / "kpi_incident_detection.csv", index=False)
    vol.round(3).to_csv(results / "kpi_volume_anomalies.csv", index=False)
    slo.round(5).to_csv(results / "slo_burndown.csv", index=False)
    cols = ["h3", "n", "median_hours", "mean_log_excess", "cell_excess_pct", "local_excess_pct", "z_naive", "q_naive",
            "gi_z", "gi_q", "gi_class", "naive_class", "true_excess_pct", "truth_class", "lat", "lon"]  # fmt: skip
    cells[cols].round(5).to_csv(results / "hotspots_h3.csv", index=False)
    eq_table.round(5).to_csv(results / "equity_by_neighbourhood.csv", index=False)
    _validation_report(results, metrics, agree, ge_raw, ge_clean, reason_summary)

    _fig_quality(daily, q_anoms, results / "quality_kpis_daily.png")
    _fig_reasons(reason_summary, results / "quarantine_by_rule.png")
    _fig_volume(df, vol, results / "volume_anomalies.png")
    _fig_slo(slo, results / "slo_burndown.png")
    _fig_equity(eq_table, results / "equity_income.png")
    hs = metrics["hotspots"]
    sub = (
        f"SYNTHETIC data. {hs['completed_requests_scored']:,} completed, contract-passing requests in "
        f"{hs['cells_scored']} H3 res-{hs['h3_resolution']} cells (n >= {hs['min_cell_n']}). Colour = Getis-Ord Gi* "
        f"on the category-adjusted log resolution time, ring-1 neighbours, Benjamini-Hochberg q < {hs['fdr_q']}."
    )
    mapviz.hotspot_map_html(cells, results / "hotspot_map.html", "311 resolution-time hot spots (H3, Gi*)", sub)
    mapviz.hotspot_map_png(cells, results / "hotspot_map.png", "311 resolution-time hot spots, Gi* (SYNTHETIC)")
    mapviz.write_geojson(cells, results / "hotspots_h3.geojson")
