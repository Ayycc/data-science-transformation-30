"""Figures written to results/figures/ (all from results/metrics.json values)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {"bm25": "#4C72B0", "dense": "#DD8452", "hybrid": "#55A868", "rerank": "#8172B3", "other": "#937860"}


def _color(name: str) -> str:
    if "rerank" in name:
        return PALETTE["rerank"]
    if name.startswith("hybrid"):
        return PALETTE["hybrid"]
    if name.startswith("dense"):
        return PALETTE["dense"]
    if name.startswith("bm25"):
        return PALETTE["bm25"]
    return PALETTE["other"]


def make_all(m: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keep = ["bm25", "dense_static", "dense_lsa", "hybrid_rrf"]
    # 1. recall@k for the structure chunking
    res = m["retrieval"]["structure"]["methods"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ks = [1, 3, 5, 10]
    for name, mm in res.items():
        if "prf" in name:
            continue
        ls = "--" if name == "dense_lsa" else "-"
        ax.plot(
            ks,
            [mm[f"recall@{k}"] for k in ks],
            marker="o",
            ls=ls,
            color=_color(name),
            label=name,
            lw=2 if name.startswith("hybrid") else 1.5,
        )
    ax.set_xticks(ks)
    ax.set_xlabel("k (retrieved clauses)")
    ax.set_ylabel("recall@k (test, in-scope)")
    ax.set_title("Retrieval quality, structure-aware chunking (SYNTHETIC corpus)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "recall_at_k_structure.png", dpi=130)
    plt.close(fig)

    # 2. recall@5 by question style and chunking
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, cname in zip(axes, ["structure", "fixed"], strict=True):
        mm = m["retrieval"][cname]["methods"]
        names = [n for n in mm if n in keep or "rerank" in n]
        x = range(len(names))
        w = 0.38
        ax.bar(
            [i - w / 2 for i in x],
            [mm[n].get("recall@5_lexical", 0) for n in names],
            w,
            label="lexical questions",
            color="#9DB4D3",
        )
        ax.bar(
            [i + w / 2 for i in x],
            [mm[n].get("recall@5_paraphrase", 0) for n in names],
            w,
            label="paraphrased questions",
            color=[_color(n) for n in names],
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels([n.replace("hybrid_rrf+rerank_", "hybrid+rerank\n") for n in names], fontsize=8)
        ax.set_title(f"{cname} chunking")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("recall@5 (test)")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["lexical questions", "paraphrased questions (coloured by method)"], loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Paraphrased questions separate the retrievers (SYNTHETIC corpus)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "recall5_by_style.png", dpi=130)
    plt.close(fig)

    # 3. faithfulness checker stress test
    s = m["faithfulness_stress_test"]["structure"]
    kinds = ["clean", "fact_swap", "citation_swap", "added_claim"]
    vals = [s["clean"]["false_alarm_rate"]] + [s[k]["detection_rate"] for k in kinds[1:]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(
        ["clean answers\n(false alarms)", "fact swapped", "citation swapped", "unsupported\nclaim added"],
        vals,
        color=["#C44E52", "#55A868", "#55A868", "#55A868"],
    )
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share flagged as unfaithful")
    ax.set_title("Faithfulness checker: injected hallucinations caught")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "faithfulness_stress_test.png", dpi=130)
    plt.close(fig)
