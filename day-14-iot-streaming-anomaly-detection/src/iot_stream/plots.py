"""Static PNG figures (matplotlib) and one interactive Plotly HTML dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from .config import DAY, HOUR  # noqa: E402

C = {"adaptive": "#2a6fdb", "no_reset": "#e08a1e", "static": "#8a8a8a", "fail": "#c62828", "warn": "#2e7d32"}
TAG = "SYNTHETIC data"


def _save(fig, path: Path) -> None:
    fig.text(0.995, 0.005, TAG, ha="right", va="bottom", fontsize=7, color="#888")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def machine_trajectory(
    win: pd.DataFrame, scores: pd.DataFrame, warns: pd.DataFrame, fails: pd.DataFrame, machine: int, thr: float, path: Path
) -> None:
    w = win[win.machine == machine].sort_values("window_end")
    s = scores[scores.machine == machine].sort_values("window_end")
    fig, ax = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
    d = w.window_end / DAY
    ax[0].plot(d, w.vib_mean, lw=1, color="#444")
    ax[0].set_ylabel("vibration\n(mm/s, 30 min mean)")
    ax[1].plot(d, w.temp_mean, lw=1, color="#444")
    ax[1].set_ylabel("temperature\n(°C)")
    ax[2].plot(s.window_end / DAY, s.hst_score, lw=1, color=C["adaptive"])
    ax[2].axhline(thr, ls="--", color="#999", lw=1)
    ax[2].set_ylabel("HST anomaly\nscore")
    ax[3].plot(s.window_end / DAY, s.rul_pred, lw=1, color=C["adaptive"], label="online RUL prediction")
    ax[3].plot(s.window_end / DAY, s.rul_true, lw=1, color="#999", ls="--", label="true RUL (capped 72 h)")
    ax[3].set_ylabel("RUL (h)")
    ax[3].legend(loc="lower left", fontsize=8, frameon=False)
    for a in ax:
        for t in fails[fails.machine == machine].failure_time:
            a.axvline(t / DAY, color=C["fail"], lw=1.5)
        for t in warns[warns.machine == machine].t:
            a.axvline(t / DAY, color=C["warn"], lw=1, ls=":")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title(f"Machine {machine}: red = failure, green dotted = warning (adaptive, combined rule)", fontsize=10, loc="left")
    ax[3].set_xlabel("day")
    _save(fig, path)


def lead_time_hist(lead: pd.DataFrame, horizon_h: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(lead.lead_h, bins=np.arange(0, horizon_h + 3, 3), color=C["adaptive"], alpha=0.85)
    med = float(lead.lead_h.median())
    ax.axvline(med, color="#222", ls="--", lw=1)
    ax.text(med, ax.get_ylim()[1] * 0.92, f" median {med:.1f} h", fontsize=9)
    ax.set_xlabel("warning lead time before failure (h)")
    ax.set_ylabel("failures")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def drift_timeline(medians: pd.DataFrame, drift: pd.DataFrame, regime_s: float, path: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for v, ls in (("no_reset", "-"),):
        m = medians[medians.variant == v]
        for ch, col in (("vib_rel", "#2a6fdb"), ("cur_rel", "#e08a1e"), ("temp_d", "#7b1fa2")):
            ax[0].plot(m.window_end / DAY, m[ch], lw=1, ls=ls, color=col, label=ch)
    ax[0].set_ylabel("fleet median residual\n(no recalibration)")
    ax[0].legend(fontsize=8, frameon=False, ncol=3)
    m = medians[medians.variant == "adaptive"]
    for ch, col in (("vib_rel", "#2a6fdb"), ("cur_rel", "#e08a1e"), ("temp_d", "#7b1fa2")):
        ax[1].plot(m.window_end / DAY, m[ch], lw=1, color=col)
    ad = drift[(drift.variant == "adaptive") & (drift.detector == "ADWIN")]
    for t in sorted(set(ad.t)):
        ax[1].axvline(t / DAY, color="#555", lw=0.6, ls=":")
    ax[1].set_ylabel("fleet median residual\n(adaptive; dotted = ADWIN reset)")
    for a in ax:
        a.axvline(regime_s / DAY, color=C["fail"], lw=1)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title("Seasonal warming (gradual) and product change on day 8 (abrupt, red line)", fontsize=10, loc="left")
    ax[1].set_xlabel("day")
    _save(fig, path)


def variant_bars(ev: pd.DataFrame, path: Path) -> None:
    e = ev[ev.rule == "combined"].set_index("variant").reindex(["adaptive", "no_reset", "static"])
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, col, lab in zip(
        ax, ("precision", "recall", "false_warnings_per_machine_day"), ("precision", "recall", "false warnings / machine-day"), strict=True
    ):
        vals = e[col].to_numpy()
        a.bar(e.index, vals, color=[C[v] for v in e.index])
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        a.set_title(lab, fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def throughput_bars(tp: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(tp.stage, tp.events_per_sec, color="#2a6fdb")
    ax.axvline(10_000, color=C["fail"], ls="--", lw=1)
    ax.text(10_000, -0.6, " 10k target", color=C["fail"], fontsize=8)
    for i, v in enumerate(tp.events_per_sec):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=8)
    ax.set_xlim(0, float(tp.events_per_sec.max()) * 1.2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.set_xlabel("raw sensor events / second (single CPU core)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def lateness_hist(lateness: np.ndarray, allowed_s: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = lateness[lateness > 0]
    ax.hist(np.log10(x), bins=80, color="#555")
    ax.axvline(np.log10(allowed_s), color=C["fail"], lw=1.2)
    ax.text(np.log10(allowed_s), ax.get_ylim()[1] * 0.9, " allowed lateness 10 min", color=C["fail"], fontsize=8)
    ax.set_xlabel("log10 lateness behind newest event (s)")
    ax.set_ylabel("events")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def dashboard(
    win: pd.DataFrame, scores: pd.DataFrame, warns: pd.DataFrame, fails: pd.DataFrame, machines: list[int], thr: float, path: Path
) -> None:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("vibration 30 min mean", "HST score", "RUL (h)")
    )
    vis = []
    for k, m in enumerate(machines):
        w = win[win.machine == m].sort_values("window_end")
        s = scores[scores.machine == m].sort_values("window_end")
        on = k == 0
        fig.add_trace(go.Scattergl(x=w.window_end / DAY, y=w.vib_mean.round(3), name=f"m{m} vib", visible=on), 1, 1)
        fig.add_trace(go.Scattergl(x=s.window_end / DAY, y=s.hst_score.round(3), name=f"m{m} hst", visible=on), 2, 1)
        fig.add_trace(go.Scattergl(x=s.window_end / DAY, y=s.rul_pred.round(1), name=f"m{m} rul pred", visible=on), 3, 1)
        fig.add_trace(
            go.Scattergl(x=s.window_end / DAY, y=s.rul_true.round(1), name=f"m{m} rul true", line={"dash": "dash"}, visible=on), 3, 1
        )
        ft = fails[fails.machine == m].failure_time / DAY
        wt = warns[warns.machine == m].t / DAY
        fig.add_trace(
            go.Scatter(
                x=ft, y=[0] * len(ft), mode="markers", marker={"symbol": "x", "size": 11, "color": C["fail"]}, name="failure", visible=on
            ),
            3,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=wt,
                y=[thr] * len(wt),
                mode="markers",
                marker={"symbol": "triangle-up", "size": 10, "color": C["warn"]},
                name="warning",
                visible=on,
            ),
            2,
            1,
        )
        vis.append(6)
    n = sum(vis)
    buttons = []
    for k, m in enumerate(machines):
        mask = [False] * n
        for j in range(6 * k, 6 * k + 6):
            mask[j] = True
        buttons.append({"label": f"machine {m}", "method": "update", "args": [{"visible": mask}]})
    fig.update_layout(
        title="Streaming predictive maintenance - SYNTHETIC fleet (adaptive variant)",
        updatemenus=[{"buttons": buttons, "x": 1.0, "y": 1.12, "xanchor": "right"}],
        height=760,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.08},
    )
    fig.update_xaxes(title_text="day", row=3, col=1)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


__all__ = [
    "machine_trajectory",
    "lead_time_hist",
    "drift_timeline",
    "variant_bars",
    "throughput_bars",
    "lateness_hist",
    "dashboard",
    "HOUR",
]
