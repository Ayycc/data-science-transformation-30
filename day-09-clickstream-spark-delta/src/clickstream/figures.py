"""Figures from results/metrics.json (all SYNTHETIC)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#2f6db5", "#e07b28", "#3a9a5b", "#8a8a8a"


def make_all(results: Path) -> list[str]:
    m = json.loads((results / "metrics.json").read_text())
    out = []
    sk = m["skew_benchmark"]["variants"]
    names = list(sk)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(names, [sk[n]["runtime_s_median"] for n in names], color=[GREY, BLUE, GREEN])
    for i, n in enumerate(names):
        ax[0].text(
            i, sk[n]["runtime_s_median"], f"{sk[n]['runtime_s_median']:.1f}s\n{sk[n]['speedup_vs_baseline']:.2f}x", ha="center", va="bottom"
        )
    ax[0].set_ylabel("wall-clock seconds (median)")
    ax[0].set_title("Skewed join runtime")
    ax[0].margins(y=0.25)
    if all(sk[n]["slowest_stage"] for n in names):
        med = [sk[n]["slowest_stage"]["median_task_ms"] / 1000 for n in names]
        mx = [sk[n]["slowest_stage"]["max_task_ms"] / 1000 for n in names]
        x = range(len(names))
        ax[1].bar([i - 0.2 for i in x], med, 0.4, label="median task", color=BLUE)
        ax[1].bar([i + 0.2 for i in x], mx, 0.4, label="slowest task", color=ORANGE)
        ax[1].set_xticks(list(x), names)
        ax[1].set_ylabel("task executor run time (s)")
        ax[1].set_title("Join stage: median vs slowest task (Spark UI)")
        ax[1].legend()
    fig.suptitle("SYNTHETIC clickstream: skew mitigation on all page views x campaign targeting")
    fig.tight_layout()
    fig.savefig(results / "skew_benchmark.png", dpi=110)
    plt.close(fig)
    out.append("skew_benchmark.png")

    lay = m["layout_benchmark"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    probes = ["user_lookup", "article_lookup", "one_day"]
    for j, (key, lab) in enumerate([("files_read", "files read"), ("runtime_s_median", "seconds (median)")]):
        b = [lay["before"][p][key] for p in probes]
        a = [lay["after"][p][key] for p in probes]
        ax[j].bar(
            [i - 0.2 for i in range(len(probes))],
            b,
            0.4,
            label=f"before OPTIMIZE ({lay['before']['files']['num_files']} files)",
            color=GREY,
        )
        ax[j].bar(
            [i + 0.2 for i in range(len(probes))],
            a,
            0.4,
            label=f"after ZORDER BY {', '.join(lay['zorder_by'])} ({lay['after']['files']['num_files']} files)",
            color=BLUE,
        )
        ax[j].set_xticks(list(range(len(probes))), probes)
        ax[j].set_ylabel(lab)
        ax[j].legend(fontsize=8)
    fig.suptitle("SYNTHETIC clickstream: silver data skipping before/after OPTIMIZE ZORDER")
    fig.tight_layout()
    fig.savefig(results / "layout_benchmark.png", dpi=110)
    plt.close(fig)
    out.append("layout_benchmark.png")

    s = pd.read_csv(results / "_session_sample.csv.gz")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(s["duration_s"] / 60, bins=60, color=BLUE)
    ax[0].set_xlabel("session duration (minutes)")
    ax[0].set_ylabel("sessions (sample)")
    ax[0].set_yscale("log")
    ax[1].hist(s["n_events"].clip(upper=60), bins=60, color=GREEN)
    ax[1].set_xlabel("events per session (clipped at 60)")
    ax[1].set_yscale("log")
    fig.suptitle("SYNTHETIC clickstream: human sessions (30-minute inactivity rule)")
    fig.tight_layout()
    fig.savefig(results / "session_distribution.png", dpi=110)
    plt.close(fig)
    out.append("session_distribution.png")

    b = m["batches"]
    ks = sorted(b, key=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(ks)
    for stage, col in [("bronze_s", GREY), ("silver_s", BLUE), ("gold_s", ORANGE)]:
        vals = [b[k][stage] for k in ks]
        ax.bar([f"batch {k}" for k in ks], vals, bottom=bottom, label=stage.replace("_s", ""), color=col)
        bottom = [x + y for x, y in zip(bottom, vals)]
    ax.set_ylabel("seconds")
    ax.set_title("SYNTHETIC clickstream: incremental batch timings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "batch_timings.png", dpi=110)
    plt.close(fig)
    out.append("batch_timings.png")
    return out
