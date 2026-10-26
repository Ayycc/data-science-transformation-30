"""Markdown report written from the run's own metrics (results/report.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _md(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(metrics: dict, mon: pd.DataFrame, reg: pd.DataFrame, out: Path) -> None:
    it = pd.DataFrame(metrics["initial_training"]["steps"])
    it["checks_passed"] = it["checks_passed"].map(lambda d: ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in d.items()))
    cols = [
        "month",
        "champion_version",
        "deviance",
        "static_deviance",
        "ae_ratio",
        "static_ae_ratio",
        "gini",
        "drift_share",
        "drifted_columns",
        "trigger",
        "suppressed_by_cooldown",
        "api_rows_per_s",
    ]
    rt = pd.DataFrame(metrics["retraining"]["retrain_steps"])
    lines = [
        "# Day 16 run report (SYNTHETIC data)",
        "",
        f"Data path: **{metrics['data']['path']}**; {metrics['scale']['rows']:,} policies over {metrics['scale']['months']} months; runtime {metrics['runtime_seconds']} s.",
        "",
        "## Initial training and gated promotion",
        "",
        _md(
            it[
                [
                    "family",
                    "note",
                    "version",
                    "action",
                    "val_tweedie_deviance",
                    "null_deviance",
                    "val_gini",
                    "val_ae_ratio",
                    "deviance_gain_vs_reference",
                    "gate_reference",
                    "checks_passed",
                ]
            ]
        ),
        "",
        "## Monthly production monitoring",
        "",
        _md(mon[cols].fillna("")),
        "",
        "## Retraining triggers",
        "",
        _md(rt) if len(rt) else "No retraining trigger fired.",
        "",
        "## Shadow / canary comparison (previous champion shadows the new one)",
        "",
        _md(pd.DataFrame(metrics["canary"])) if metrics["canary"] else "No shadow deployment ran.",
        "",
        "## Model registry",
        "",
        _md(reg.drop(columns=["run_id"], errors="ignore").fillna("")),
        "",
        "## Exported champion API smoke test",
        "",
        "\n".join(f"- {k}: {v}" for k, v in metrics["api_smoke_exported_model"].items()),
        "",
    ]
    out.write_text("\n".join(lines))
