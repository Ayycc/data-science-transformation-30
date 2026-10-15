"""Write results/: metrics.json, CSV tables, figures (PNG + Plotly HTML) and report.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .backtest import SELECTED, config_name  # noqa: E402

# Reference categorical palette (fixed order) and text tokens.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def figures(res: dict, df: pd.DataFrame, out: Path) -> list[str]:
    files = []
    s = res["summary"]

    # 1. MASE by model (horizontal bars, sorted).
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ss = s.sort_values("mase", ascending=False)
    naive = res["best_naive"]
    colors = [C[0] if m == SELECTED else (C[1] if m == naive else "#a9a8a2") for m in ss["model"]]
    ax.barh(ss["model"], ss["mase"], color=colors, height=0.6)
    for y, v in enumerate(ss["mase"]):
        ax.text(v + 0.01, y, _fmt(v), va="center", color=INK, fontsize=9)
    ax.set_xlabel("MASE (weekly seasonal-naive scale; lower is better)")
    ax.set_title("Day-ahead point accuracy, 12-fold rolling-origin backtest", color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "mase_by_model.png", dpi=130)
    plt.close(fig)
    files.append("mase_by_model.png")

    # 2. Coverage heatmap region x horizon for the selected model (diverging around 90%).
    cov = res["cov_region_horizon"]
    h = cov[cov["model"] == SELECTED].pivot(index="region", columns="horizon", values="coverage")
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f1f0ec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(h.to_numpy() * 100, aspect="auto", cmap=cmap, vmin=80, vmax=100)
    ax.set_yticks(range(len(h.index)), h.index)
    ax.set_xticks(range(0, 24, 2), [str(c) for c in h.columns[::2]])
    ax.set_xlabel("Horizon (hours after the 23:00 forecast origin)")
    ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.to_numpy()[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Empirical coverage, %")
    ax.set_title(f"90% interval coverage by region and horizon: {SELECTED}", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "coverage_region_horizon.png", dpi=130)
    plt.close(fig)
    files.append("coverage_region_horizon.png")

    # 3. Coverage by fold for the interval methods.
    cf = res["cov_fold"]
    show = [SELECTED, f"LightGBM+SplitCP static {res['cal_days']}d", "MSTL", "AutoARIMA", naive]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(show):
        g = cf[cf["model"] == m]
        ax.plot(g["fold"] + 1, g["coverage"] * 100, marker="o", ms=4, lw=2 if m == SELECTED else 1.4,
                color=C[i], label=m)  # fmt: skip
    ax.axhspan(87, 93, color=GRID, alpha=0.7, lw=0, label="87-93% target band")
    ax.axhline(90, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(1, len(cf["fold"].unique()) + 1))
    ax.set_xlabel("Backtest fold (28 days each, Jan 31 to Dec 31 2024)")
    ax.set_ylabel("Coverage of nominal 90% interval, %")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Interval calibration per fold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "coverage_by_fold.png", dpi=130)
    plt.close(fig)
    files.append("coverage_by_fold.png")

    # 4. Example forecast with interval: CENTRAL around the regime shift, and SOUTH in July (Plotly + PNG).
    pred = res["pred"]
    sel = pred[pred["model"] == SELECTED]
    windows = [("CENTRAL", "2024-03-25", "2024-04-22"), ("SOUTH", "2024-07-15", "2024-08-05")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    pfig = go.Figure()
    for ax, (region, a, b) in zip(axes, windows):
        g = sel[(sel["region"] == region) & (sel["ds"] >= a) & (sel["ds"] < b)].sort_values("ds")
        if g.empty:
            continue
        ax.fill_between(g["ds"], g["q05"], g["q95"], color=C[0], alpha=0.2, lw=0, label="90% conformal interval")
        ax.plot(g["ds"], g["q50"], color=C[0], lw=1.2, label="Day-ahead forecast")
        ax.plot(g["ds"], g["y"], color=INK, lw=0.9, label="Actual (synthetic)")
        miss = g[(g["y"] < g["q05"]) | (g["y"] > g["q95"])]
        ax.scatter(miss["ds"], miss["y"], s=10, color=C[7], zorder=3, label="Outside interval")
        ax.set_ylabel(f"{region} load, MW")
        ax.set_title(f"{region}, {a} to {b}", color=INK, loc="left", fontsize=10)
        for name, col in (("q95", "upper"), ("q05", "lower")):
            pfig.add_trace(
                go.Scatter(x=g["ds"], y=g[name], name=f"{region} {col} 90%", line=dict(width=0.5, color=C[0]),
                           legendgroup=region, showlegend=False)
            )  # fmt: skip
        pfig.add_trace(go.Scatter(x=g["ds"], y=g["q50"], name=f"{region} forecast", line=dict(color=C[0], width=1.5),
                                  legendgroup=region))  # fmt: skip
        pfig.add_trace(go.Scatter(x=g["ds"], y=g["y"], name=f"{region} actual (synthetic)",
                                  line=dict(color=INK, width=1), legendgroup=region))  # fmt: skip
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "forecast_examples.png", dpi=130)
    plt.close(fig)
    files.append("forecast_examples.png")
    pfig.update_layout(
        title="Day-ahead forecasts with 90% conformal intervals (SYNTHETIC data)",
        hovermode="x unified",
        template="plotly_white",
        yaxis_title="MW",
    )
    pfig.write_html(out / "forecast_examples.html", include_plotlyjs="cdn", full_html=True)
    files.append("forecast_examples.html")

    # 5. Coverage vs width: one row per model, coverage bar with the target band, width as a label.
    ss = s.sort_values("coverage90")
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(ss) + 1.4))
    ax.axvspan(87, 93, color=GRID, alpha=0.8, lw=0, label="87-93% target band")
    colors = [C[0] if m == SELECTED else "#a9a8a2" for m in ss["model"]]
    ax.barh(ss["model"], ss["coverage90"] * 100, color=colors, height=0.6)
    for y, (c, w) in enumerate(zip(ss["coverage90"], ss["mean_width_pct_of_load"])):
        ax.text(c * 100 + 0.4, y, f"{c * 100:.1f}%  (width {w:.1f}% of load)", va="center", fontsize=8, color=INK)
    ax.set_xlim(50, 118)
    ax.axvline(90, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Empirical coverage of the nominal 90% interval, %")
    ax.grid(axis="y", visible=False)
    ax.set_title("Calibration and sharpness by model", color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_width.png", dpi=130)
    plt.close(fig)
    files.append("coverage_vs_width.png")
    return files


def acceptance(res: dict) -> dict:
    s = res["summary"].set_index("model")
    sel = s.loc[SELECTED]
    cov_r = res["cov_region"]
    cr = cov_r[cov_r["model"] == SELECTED].set_index("region")["coverage"]
    crh = res["cov_region_horizon"]
    crh = crh[crh["model"] == SELECTED]["coverage"]
    best_model = s["mase"].idxmin()
    naive = s.loc[res["best_naive"], "mase"]
    improvement = (1 - s.loc[best_model, "mase"] / naive) * 100
    return {
        "coverage_overall": float(sel["coverage90"]),
        "coverage_overall_in_87_93": bool(0.87 <= sel["coverage90"] <= 0.93),
        "coverage_min_region": float(cr.min()),
        "coverage_max_region": float(cr.max()),
        "all_regions_in_87_93": bool(((cr >= 0.87) & (cr <= 0.93)).all()),
        "region_horizon_cells": int(len(crh)),
        "region_horizon_cells_in_87_93": int(((crh >= 0.87) & (crh <= 0.93)).sum()),
        "region_horizon_min": float(crh.min()),
        "region_horizon_max": float(crh.max()),
        "best_model": best_model,
        "best_seasonal_naive": res["best_naive"],
        "best_model_mase": float(s.loc[best_model, "mase"]),
        "seasonal_naive_mase": float(naive),
        "mase_improvement_vs_seasonal_naive_pct": float(improvement),
        "beats_seasonal_naive_by_15pct": bool(improvement >= 15),
    }


def _md_table(df: pd.DataFrame, cols: list[str], fmts: dict) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmts.get(c, str)(r[c]) for c in cols) + " |")
    return head + "\n".join(rows) + "\n"


def write_results(res: dict, df: pd.DataFrame, manifest: dict, out: Path, runtime: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    acc = acceptance(res)
    s = res["summary"]
    s.to_csv(out / "summary_by_model.csv", index=False, float_format="%.5f")
    res["per_fold"].to_csv(out / "metrics_by_fold.csv", index=False, float_format="%.5f")
    res["per_region"].to_csv(out / "metrics_by_region.csv", index=False, float_format="%.5f")
    res["cov_region_horizon"].to_csv(out / "coverage_region_horizon.csv", index=False, float_format="%.5f")
    res["val_table"].to_csv(out / "conformal_selection_validation.csv", index=False, float_format="%.5f")
    if not res["stress"].empty:
        res["stress"].to_csv(out / "stress_test.csv", index=False, float_format="%.4f")
    files = figures(res, df, out)

    metrics = {
        "data": {k: v for k, v in manifest.items() if k != "file"},
        "backtest": {
            "folds": len(res["fold_info"]),
            "fold_days": res["fold_days"],
            "horizon_hours": 24,
            "forecast_origin": "daily at 23:00 for hours 00:00-23:00 of the next day",
            "test_period": [res["fold_info"][0]["start"], "2024-12-31"] if res["scale"] == "full" else None,
            "points_per_model": int(s["n_points"].iloc[0]),
            "fold_info": res["fold_info"],
        },
        "conformal_selection": {
            "selected": config_name(*res["best_cfg"]),
            "criterion": "mean |coverage - 0.90| over region x horizon cells on validation folds before the test year, "
            "then width",
            "validation": res["val_table"].to_dict(orient="records"),
        },
        "models": s.to_dict(orient="records"),
        "acceptance": acc,
        "stress_test": res["stress"].to_dict(orient="records"),
        "runtime_s": runtime,
        "figures": files,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    (out / "report.md").write_text(report_md(res, acc, manifest))
    return metrics


def report_md(res: dict, acc: dict, manifest: dict) -> str:
    s = res["summary"]
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    f2 = lambda x: f"{x:.2f}"  # noqa: E731
    f0 = lambda x: f"{x:,.0f}"  # noqa: E731
    sgn = lambda x: f"{x:+.1f}%"  # noqa: E731
    lines = [
        "# Day 04 backtest report (SYNTHETIC data)",
        "",
        f"Generated by `make run`. Data path: **{manifest['path_used']}**; public probe reachable: "
        f"{manifest['public_probe'].get('reachable')}. {len(res['fold_info'])} rolling-origin folds of "
        f"{res['fold_days']} days, day-ahead (24 h) forecasts issued daily at 23:00.",
        "",
        "## Models",
        "",
        _md_table(
            s,
            [
                "model",
                "mase",
                "mase_vs_best_seasonal_naive_pct",
                "smape_pct",
                "scaled_pinball",
                "pinball_mw",
                "coverage90",
                "mean_width_pct_of_load",
                "winkler_mw",
            ],  # fmt: skip
            {
                "mase": f3,
                "mase_vs_best_seasonal_naive_pct": sgn,
                "smape_pct": f2,
                "scaled_pinball": f3,
                "pinball_mw": f2,
                "coverage90": pct,
                "mean_width_pct_of_load": f2,
                "winkler_mw": f0,
            },  # fmt: skip
        ),
        "",
        "## Acceptance",
        "",
        "```json",
        json.dumps(acc, indent=2),
        "```",
        "",
        "## Conformal configuration chosen on validation folds (before the test year)",
        "",
        _md_table(
            res["val_table"],
            [
                "config",
                "coverage",
                "cell_cov_abs_err",
                "region_cov_abs_err",
                "min_region_cov",
                "max_region_cov",
                "width_pct",
            ],  # fmt: skip
            {
                "coverage": pct,
                "cell_cov_abs_err": f3,
                "region_cov_abs_err": f3,
                "min_region_cov": pct,
                "max_region_cov": pct,
                "width_pct": f2,
            },  # fmt: skip
        ),
        "",
        "## Coverage by region (test folds)",
        "",
    ]
    cr = res["cov_region"].pivot(index="model", columns="region", values="coverage").reset_index()
    regions = [c for c in cr.columns if c != "model"]
    lines.append(_md_table(cr, ["model", *regions], {r: pct for r in regions}))
    lines += ["", "## Coverage by fold (test folds)", ""]
    cf = res["cov_fold"].pivot(index="model", columns="fold", values="coverage").reset_index()
    folds = [c for c in cf.columns if c != "model"]
    cf.columns = ["model", *[str(int(c) + 1) for c in folds]]
    lines.append(_md_table(cf, list(cf.columns), {c: pct for c in cf.columns if c != "model"}))
    sel_f = res["per_fold"][res["per_fold"]["model"] == SELECTED]
    lines += ["", f"## {SELECTED} by fold", ""]
    lines.append(
        _md_table(
            sel_f,
            ["fold", "start", "mase", "smape_pct", "coverage90", "mean_width_pct_of_load"],
            {
                "fold": lambda x: str(int(x) + 1),
                "mase": f3,
                "smape_pct": f2,
                "coverage90": pct,
                "mean_width_pct_of_load": f2,
            },
        )  # fmt: skip
    )
    pr = res["per_region"][res["per_region"]["model"].isin([SELECTED, res["best_naive"], "MSTL"])]
    lines += ["", "## By region", ""]
    lines.append(
        _md_table(
            pr,
            ["region", "model", "mase", "smape_pct", "coverage90", "mean_width_pct_of_load"],
            {"mase": f3, "smape_pct": f2, "coverage90": pct, "mean_width_pct_of_load": f2},
        )  # fmt: skip
    )
    if not res["stress"].empty:
        st = res["stress"]
        lines += ["", "## Temperature stress test (stretch)", ""]
        lines.append(
            _md_table(
                st,
                ["scenario", "world", "region", "actual_peak_mw", "forecast_peak_mw", "mape_pct", "coverage90"],
                {"actual_peak_mw": f0, "forecast_peak_mw": f0, "mape_pct": f2, "coverage90": pct},
            )  # fmt: skip
        )
    lines.append("")
    return "\n".join(lines)
