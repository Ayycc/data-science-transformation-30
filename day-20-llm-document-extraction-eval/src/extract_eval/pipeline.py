"""End-to-end run: corpus -> pdfplumber -> extractors -> validation gate -> evaluation -> results/."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .downstream import DownstreamStore
from .evaluate import (
    IsotonicCalibrator,
    choose_threshold,
    ece,
    field_rows,
    match_line_items,
    per_field_table,
    prf,
    reliability,
    routing_summary,
)
from .extractors.base import ExtractionResult
from .extractors.rules import RulesExtractor
from .generator import SEED, generate, write_corpus
from .pdf_text import DocText, read_pdf
from .prompts import registry
from .schemas import validate_record

ROOT = Path(__file__).resolve().parents[2]
TARGET_ACCURACY = 0.99
SEEN = {"msa_prose", "cover_sheet", "procurement_letter", "ruled_table", "text_table", "compact_lines"}


def build_corpus(n_docs: int, raw_dir: Path, sample_dir: Path, n_sample: int = 30) -> dict[str, Any]:
    docs = generate(n_docs, SEED)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    write_corpus(docs, raw_dir)
    if sample_dir is not None:
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        write_corpus(docs[:n_sample], sample_dir)
    return {
        "path_used": "synthetic_generator",
        "n_docs": len(docs),
        "seed": SEED,
        "note": "Prompt specifies synthetic contracts/invoices; no download attempted, no network calls.",
    }


def load_corpus(corpus_dir: Path) -> tuple[list[DocText], dict[str, dict[str, Any]]]:
    golds = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((corpus_dir / "gold").glob("*.json"))}
    texts = [read_pdf(corpus_dir / "pdf" / f"{d}.pdf") for d in golds]
    return texts, golds


def make_split(golds: dict[str, dict[str, Any]], dev_frac: float = 0.4, seed: int = SEED) -> dict[str, str]:
    rng = random.Random(seed + 1)
    split: dict[str, str] = {}
    for t in ("contract", "invoice"):
        ids = sorted(d for d, g in golds.items() if g["label"]["doc_type"] == t)
        rng.shuffle(ids)
        k = round(len(ids) * dev_frac)
        split.update({d: ("dev" if i < k else "test") for i, d in enumerate(ids)})
    return split


def run_extractor(ex: Any, texts: list[DocText]) -> list[ExtractionResult]:
    out = []
    for dt in texts:
        r = ex.extract(dt)
        model, errors = validate_record(r.doc_type, r.record_payload())
        r.valid, r.validation_errors = model is not None, errors
        out.append(r)
    return out


def _pct(x: list[float], q: float) -> float:
    return float(np.percentile(x, q)) if x else 0.0


def evaluate_all(
    results: dict[str, list[ExtractionResult]],
    texts: list[DocText],
    golds: dict[str, dict[str, Any]],
    split: dict[str, str],
    res_dir: Path,
) -> tuple[dict[str, Any], dict[str, tuple[IsotonicCalibrator, float]]]:
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat([field_rows(r, golds, split) for r in results.values()], ignore_index=True)
    all_rows["seen_template"] = all_rows["template"].isin(SEEN)
    parse_s = {t.doc_id: t.parse_seconds for t in texts}
    metrics: dict[str, Any] = {"extractors": {}}
    calibrated: list[pd.DataFrame] = []
    routers: dict[str, tuple[IsotonicCalibrator, float]] = {}
    tables: list[pd.DataFrame] = []
    for name, res in results.items():
        rows = all_rows[all_rows["extractor"] == name].copy()
        dev, test = rows[rows["split"] == "dev"].copy(), rows[rows["split"] == "test"].copy()
        cal = IsotonicCalibrator().fit(dev)
        dev["conf_cal"], test["conf_cal"] = cal.transform(dev), cal.transform(test)
        thr = choose_threshold(dev["conf_cal"].to_numpy(), dev["correct"].to_numpy(), TARGET_ACCURACY)
        routing = routing_summary(test, "conf_cal", thr)
        routers[name] = (cal, thr)
        calibrated.append(test)
        tables.append(per_field_table(test))
        test_ids = {d for d, s in split.items() if s == "test"}
        rtest = [r for r in res if r.doc_id in test_ids]
        # line items (invoices)
        li = {"tp": 0, "tp_strict": 0, "n_pred": 0, "n_gold": 0}
        for r in rtest:
            g = golds[r.doc_id]["label"]
            if g["doc_type"] == "invoice":
                m = match_line_items(r.line_items if r.doc_type == "invoice" else [], g["line_items"])
                for k in li:
                    li[k] += m[k]
        doc_acc = test.groupby("doc_id")["correct"].all()
        lat = [r.latency_s for r in rtest]
        tot = [r.latency_s + parse_s[r.doc_id] for r in rtest]
        by_slice = {}
        for label, mask in {
            "seen_clean": test["seen_template"] & ~test["scanned"],
            "seen_scanned": test["seen_template"] & test["scanned"],
            "unseen_clean": ~test["seen_template"] & ~test["scanned"],
            "unseen_scanned": ~test["seen_template"] & test["scanned"],
        }.items():
            by_slice[label] = {
                "field_accuracy": float(test.loc[mask, "correct"].mean()) if mask.any() else None,
                "n_docs": int(test.loc[mask, "doc_id"].nunique()),
            }
        metrics["extractors"][name] = {
            "test_docs": len(rtest),
            "field_accuracy_fuzzy": float(test["correct"].mean()),
            "field_accuracy_exact": float(test["exact"].mean()),
            "doc_all_fields_correct": float(doc_acc.mean()),
            "schema_valid_rate": float(np.mean([r.valid for r in rtest])),
            "doc_type_routing_accuracy_all_docs": float(
                np.mean([r.doc_type == golds[r.doc_id]["label"]["doc_type"] for r in res])
            ),
            "line_items": {
                **li,
                "loose": prf(li["tp"], li["n_pred"], li["n_gold"]),
                "strict": prf(li["tp_strict"], li["n_pred"], li["n_gold"]),
            },
            "by_slice": by_slice,
            "calibration_test": {
                "ece_raw": ece(test["confidence"], test["correct"]),
                "ece_calibrated": ece(test["conf_cal"], test["correct"]),
                "brier_raw": float(np.mean((test["confidence"] - test["correct"]) ** 2)),
                "brier_calibrated": float(np.mean((test["conf_cal"] - test["correct"]) ** 2)),
            },
            "routing_test": routing,
            "cost_latency": {
                "extract_p50_ms": _pct(lat, 50) * 1e3,
                "extract_p95_ms": _pct(lat, 95) * 1e3,
                "end_to_end_p50_ms": _pct(tot, 50) * 1e3,
                "end_to_end_p95_ms": _pct(tot, 95) * 1e3,
                "mean_input_tokens": float(np.mean([r.input_tokens for r in rtest])),
                "mean_output_tokens": float(np.mean([r.output_tokens for r in rtest])),
                "cost_per_doc_usd": float(np.mean([r.cost_usd for r in rtest])),
                "total_cost_usd": float(sum(r.cost_usd for r in rtest)),
                "mean_attempts": float(np.mean([r.attempts for r in rtest])),
            },
            "error_taxonomy_test": test["error_category"].value_counts().to_dict(),
        }
        metrics["extractors"][name]["active_sampling"] = active_sampling(test, [r for r in rtest])
    cal_df = pd.concat(calibrated, ignore_index=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(res_dir / "per_field_metrics.csv", index=False, float_format="%.4f")
    (res_dir / "per_field_metrics.md").write_text(per_field_markdown(table), encoding="utf-8")
    cal_df.drop(columns=["pred", "gold"]).to_csv(res_dir / "field_predictions_test.csv", index=False, float_format="%.4f")
    write_error_analysis(cal_df, res_dir / "error_analysis.md", metrics)
    make_figures(cal_df, table, metrics, fig_dir)
    return metrics, routers


def active_sampling(
    test: pd.DataFrame, rtest: list[ExtractionResult], budgets: tuple[float, ...] = (0.1, 0.2, 0.3)
) -> dict[str, Any]:
    """Pick documents for human review by expected error count (sum of 1 - calibrated confidence; invalid first)."""
    valid = {r.doc_id: r.valid for r in rtest}
    per_doc = test.groupby("doc_id").agg(
        exp_err=("conf_cal", lambda c: float((1 - c).sum())), errors=("correct", lambda c: int((~c).sum()))
    )
    per_doc["invalid"] = [not valid[d] for d in per_doc.index]
    ranked = per_doc.sort_values(["invalid", "exp_err"], ascending=[False, False])
    total = int(per_doc["errors"].sum())
    rng = np.random.default_rng(SEED)
    out = {}
    for b in budgets:
        k = int(round(b * len(per_doc)))
        found = int(ranked["errors"].iloc[:k].sum())
        rand = float(np.mean([per_doc["errors"].to_numpy()[rng.permutation(len(per_doc))[:k]].sum() for _ in range(500)]))
        out[f"{int(b * 100)}pct_docs"] = {
            "docs_reviewed": k,
            "errors_found_active": found,
            "errors_found_random_mean": rand,
            "share_found_active": found / max(1, total),
            "share_found_random": rand / max(1, total),
        }
    out["total_field_errors"] = total
    return out


def per_field_markdown(t: pd.DataFrame) -> str:
    lines = [
        "# Per-field metrics (test split, SYNTHETIC documents)",
        "",
        "exact = normalised exact match; fuzzy = rapidfuzz token_sort_ratio >= 90 for party names and governing law (other fields: exact).",
        "coverage = share of documents with a non-null prediction; acc_clean / acc_scanned split by scanned-style noise.",
        "",
    ]
    for ex, g in t.groupby("extractor", sort=False):
        lines += [
            f"## {ex}",
            "",
            "| doc type | field | n | exact | fuzzy | coverage | acc clean | acc scanned | mean conf |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in g.iterrows():
            lines.append(
                f"| {r.doc_type} | {r.field} | {int(r.n)} | {r.exact_acc:.3f} | {r.fuzzy_acc:.3f} | {r.coverage:.3f} | {r.acc_clean:.3f} | {r.acc_scanned:.3f} | {r.mean_conf:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_error_analysis(df: pd.DataFrame, path: Path, metrics: dict[str, Any]) -> None:
    errs = df[~df["correct"]]
    lines = [
        "# Error analysis (test split, SYNTHETIC documents)",
        "",
        "Each wrong field is assigned the first matching category:",
        "",
        "| category | rule |",
        "|---|---|",
        "| distractor_confusion | prediction equals a decoy value planted in the document (liability cap, superseded date, order date, balance forward, termination notice) |",
        "| number_format | amount off by a power of ten (decimal/thousands separator confusion) |",
        "| date_format | day and month swapped |",
        "| unseen_layout | document uses a template the rules were not written for (held-out layouts) |",
        "| ocr_noise | wrong or missing value on a scanned-style document |",
        "| derived_value_error | field had to be computed (expiry from term, due date from terms) and is wrong |",
        "| entity_normalisation | party name printed in a long-suffix variant and not normalised |",
        "| other | anything else |",
        "",
        "Each error also has a mode: missed (null predicted, value expected), spurious (value predicted, null expected) or wrong_value.",
        "",
    ]
    for ex, g in errs.groupby("extractor", sort=False):
        n_fields = int((df["extractor"] == ex).sum())
        lines += [
            f"## {ex}: {len(g)} wrong fields of {n_fields}",
            "",
            "| category | count | share of errors | missed / spurious / wrong | top fields |",
            "|---|---|---|---|---|",
        ]
        for cat, gc in sorted(g.groupby("error_category"), key=lambda kv: -len(kv[1])):
            top = ", ".join(f"{f} ({n})" for f, n in gc["field"].value_counts().head(3).items())
            modes = gc["error_mode"].value_counts()
            mm = " / ".join(str(int(modes.get(k, 0))) for k in ("missed", "spurious", "wrong_value"))
            lines.append(f"| {cat} | {len(gc)} | {len(gc) / len(g):.1%} | {mm} | {top} |")
        tmpl = g.groupby("template").size().sort_values(ascending=False)
        lines += [
            "",
            "Errors by template: " + ", ".join(f"{k} {v}" for k, v in tmpl.items()),
            "",
            "Examples (first two per category):",
            "",
            "| doc | template | field | gold | predicted | category |",
            "|---|---|---|---|---|---|",
        ]
        for cat, gc in g.groupby("error_category"):
            for _, r in gc.head(2).iterrows():
                lines.append(f"| {r.doc_id} | {r.template} | {r.field} | {r.gold} | {r.pred} | {cat} |")
        rt = metrics["extractors"][ex]["routing_test"]
        lines += [
            "",
            f"Human-in-the-loop routing (threshold {rt['threshold']:.3f} on calibrated confidence, chosen on dev for >= {TARGET_ACCURACY:.0%} auto-accept accuracy): "
            f"{rt['errors_caught_by_review']} of {rt['errors_total']} wrong fields are sent to review; {rt['errors_leaked']} are auto-accepted.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(df: pd.DataFrame, table: pd.DataFrame, metrics: dict[str, Any], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua, ink, ink2, grid = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update(
        {
            "axes.edgecolor": ink2,
            "axes.labelcolor": ink,
            "xtick.color": ink2,
            "ytick.color": ink2,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )
    colors = [blue, orange, aqua]
    names = list(metrics["extractors"])

    # 1. per-field accuracy
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, dt in zip(axes, ["contract", "invoice"], strict=True):
        t = table[table["doc_type"] == dt]
        fields = list(dict.fromkeys(t["field"]))
        y = np.arange(len(fields))
        h = 0.8 / len(names)
        for k, ex in enumerate(names):
            v = t[t["extractor"] == ex].set_index("field").reindex(fields)["fuzzy_acc"]
            ax.barh(y + k * h, v, height=h * 0.9, color=colors[k % 3], label=ex)
        ax.set_yticks(y + h * (len(names) - 1) / 2, fields)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        ax.set_title(f"{dt}s: field accuracy (test)", loc="left", color=ink)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=len(names))
    fig.text(0.01, 0.01, "SYNTHETIC documents", color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(fig_dir / "per_field_accuracy.png", dpi=130)
    plt.close(fig)

    # 2. reliability diagram (raw vs calibrated)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), squeeze=False)
    for ax, ex in zip(axes[0], names, strict=True):
        d = df[df["extractor"] == ex]
        ax.plot([0, 1], [0, 1], color=ink2, lw=1, ls="--")
        for col, c, lab in (("confidence", orange, "raw"), ("conf_cal", blue, "isotonic (fit on dev)")):
            rel = reliability(d[col], d["correct"])
            ax.plot(
                rel["mean_conf"],
                rel["accuracy"],
                marker="o",
                ms=5,
                lw=2,
                color=c,
                label=f"{lab}, ECE {ece(d[col], d['correct']):.3f}",
            )
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("observed accuracy")
        ax.set_title(ex, loc="left")
        ax.grid(color=grid)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=130)
    plt.close(fig)

    # 3. routing trade-off for each extractor
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, ex in enumerate(names):
        d = df[(df["extractor"] == ex)]
        ok = d["valid_record"].to_numpy()
        ts = np.linspace(0, 1, 101)
        auto = [((d["conf_cal"] >= t) & ok).mean() for t in ts]
        acc = [d.loc[(d["conf_cal"] >= t) & ok, "correct"].mean() if ((d["conf_cal"] >= t) & ok).any() else np.nan for t in ts]
        ax.plot(auto, acc, lw=2, color=colors[k % 3], label=ex)
        rt = metrics["extractors"][ex]["routing_test"]
        if rt["auto_accepted_accuracy"] is not None:
            ax.plot(rt["auto_accept_rate"], rt["auto_accepted_accuracy"], "o", ms=8, color=colors[k % 3], mec="#fcfcfb", mew=2)
    ax.axhline(TARGET_ACCURACY, color=ink2, lw=1, ls="--")
    ax.text(0.01, TARGET_ACCURACY + 0.002, f"target {TARGET_ACCURACY:.0%}", color=ink2, fontsize=8)
    ax.set_xlabel("share of fields auto-accepted")
    ax.set_ylabel("accuracy of auto-accepted fields")
    ax.set_title("HITL routing: automation vs accuracy (test; dot = chosen threshold)", loc="left", fontsize=9)
    ax.grid(color=grid)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "routing_tradeoff.png", dpi=130)
    plt.close(fig)

    # 4. failure taxonomy
    errs = df[~df["correct"]]
    cats = errs["error_category"].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(cats))
    h = 0.8 / len(names)
    for k, ex in enumerate(names):
        v = errs[errs["extractor"] == ex]["error_category"].value_counts().reindex(cats).fillna(0)
        ax.barh(y + k * h, v, height=h * 0.9, color=colors[k % 3], label=ex)
    ax.set_yticks(y + h * (len(names) - 1) / 2, cats)
    ax.invert_yaxis()
    ax.set_xlabel("wrong fields (test split)")
    ax.grid(axis="x", color=grid)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Failure taxonomy", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_taxonomy.png", dpi=130)
    plt.close(fig)


def write_report(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# Day 20 run report (SYNTHETIC data)",
        "",
        f"Data path: `{metrics['data']['path_used']}`, {metrics['data']['n_docs']} documents, "
        f"split {metrics['split']['dev']} dev / {metrics['split']['test']} test. LLM: {metrics['llm']['status']}.",
        "",
        "| extractor | field acc (fuzzy) | field acc (exact) | docs fully correct | schema-valid | line-item F1 | ECE raw -> cal | auto-accept | auto-accepted acc | p50 ms/doc | $/doc |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ex, m in metrics["extractors"].items():
        c, r, cl = m["calibration_test"], m["routing_test"], m["cost_latency"]
        aa = f"{r['auto_accepted_accuracy']:.3f}" if r["auto_accepted_accuracy"] is not None else "n/a"
        lines.append(
            f"| {ex} | {m['field_accuracy_fuzzy']:.3f} | {m['field_accuracy_exact']:.3f} | {m['doc_all_fields_correct']:.3f} | {m['schema_valid_rate']:.3f} | "
            f"{m['line_items']['loose']['f1']:.3f} | {c['ece_raw']:.3f} -> {c['ece_calibrated']:.3f} | {r['auto_accept_rate']:.3f} | {aa} | "
            f"{cl['end_to_end_p50_ms']:.1f} | {cl['cost_per_doc_usd']:.4f} |"
        )
    lines += [
        "",
        "## Accuracy by slice (test)",
        "",
        "| extractor | seen clean | seen scanned | unseen clean | unseen scanned |",
        "|---|---|---|---|---|",
    ]
    for ex, m in metrics["extractors"].items():
        s = m["by_slice"]
        lines.append(
            f"| {ex} | "
            + " | ".join(
                f"{s[k]['field_accuracy']:.3f} (n={s[k]['n_docs']})" if s[k]["field_accuracy"] is not None else "n/a"
                for k in ("seen_clean", "seen_scanned", "unseen_clean", "unseen_scanned")
            )
            + " |"
        )
    lines += [
        "",
        "## Active sampling for review (test)",
        "",
        "| extractor | budget | docs | errors found (active) | errors found (random, mean of 500) |",
        "|---|---|---|---|---|",
    ]
    for ex, m in metrics["extractors"].items():
        a = m["active_sampling"]
        for k, v in a.items():
            if isinstance(v, dict):
                lines.append(
                    f"| {ex} | {k} | {v['docs_reviewed']} | {v['errors_found_active']} ({v['share_found_active']:.1%}) | {v['errors_found_random_mean']:.1f} ({v['share_found_random']:.1%}) |"
                )
    lines += ["", "## Validation gate", "", f"{json.dumps(metrics['downstream'], indent=1)}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    n_docs: int = 500, res_dir: Path | None = None, data_dir: Path | None = None, write_sample: bool = True
) -> dict[str, Any]:
    t_start = time.perf_counter()
    res_dir = res_dir or ROOT / "results"
    data_dir = data_dir or ROOT / "data"
    res_dir.mkdir(parents=True, exist_ok=True)
    data_info = build_corpus(n_docs, data_dir / "raw", data_dir / "sample" if write_sample else None)
    texts, golds = load_corpus(data_dir / "raw")
    split = make_split(golds)
    extractors: list[Any] = [RulesExtractor("strict"), RulesExtractor("rules")]
    llm_status = "skipped: ANTHROPIC_API_KEY not set (offline rules extractors only)"
    llm_texts = texts
    from .extractors.llm import ClaudeExtractor

    if ClaudeExtractor.available():
        cx = ClaudeExtractor()
        cap = int(os.environ.get("LLM_MAX_DOCS", "0") or 0)
        llm_status = f"live: model={cx.model}, effort={cx.effort}, prompt={cx.prompt_version}" + (
            f", first {cap} docs" if cap else ""
        )
    results: dict[str, list[ExtractionResult]] = {}
    for ex in extractors:
        results[ex.name] = run_extractor(ex, texts)
    if ClaudeExtractor.available():
        if cap:
            llm_texts = texts[:cap]
        results["claude"] = run_extractor(cx, llm_texts)
    if len(llm_texts) != len(texts):  # compare like with like when the LLM ran on a subset
        keep = {t.doc_id for t in llm_texts}
        results = {k: [r for r in v if r.doc_id in keep] for k, v in results.items()}
    metrics: dict[str, Any] = {
        "data": data_info,
        "split": {s: sum(v == s for v in split.values()) for s in ("dev", "test")},
        "llm": {"status": llm_status},
        "prompt_registry": registry(),
    }
    ev, routers = evaluate_all(results, texts, golds, split, res_dir)
    # validation gate: every extractor's output passes through it; only valid records reach accepted.jsonl,
    # and each accepted record carries the fields the HITL rule sends to a reviewer (calibrated conf < threshold)
    down: dict[str, Any] = {}
    for name, res in results.items():
        cal, thr = routers[name]
        store = DownstreamStore(res_dir / "downstream" / name)
        for r in res:
            conf = pd.DataFrame({"field": list(r.fields), "confidence": [r.confidence.get(f, 0.0) for f in r.fields]})
            review = [f for f, c in zip(conf["field"], cal.transform(conf), strict=True) if c < thr]
            store.route(r, review)
        lines = store.accepted_path.read_text(encoding="utf-8").splitlines()
        down[name] = {
            "accepted": store.n_accepted,
            "quarantined": store.n_quarantined,
            "accepted_with_no_review_fields": sum(1 for ln in lines if not json.loads(ln)["fields_needing_review"]),
            "invalid_written_downstream": sum(
                1 for ln in lines if validate_record(json.loads(ln)["record"]["doc_type"], json.loads(ln)["record"])[0] is None
            ),
        }
    metrics["downstream"] = down
    metrics.update(ev)
    metrics["pdf_parse"] = {
        "p50_ms": _pct([t.parse_seconds for t in texts], 50) * 1e3,
        "p95_ms": _pct([t.parse_seconds for t in texts], 95) * 1e3,
    }
    metrics["runtime"] = {
        "seconds": time.perf_counter() - t_start,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    write_report(metrics, res_dir / "report.md")
    return metrics
