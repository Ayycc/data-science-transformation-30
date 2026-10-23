"""End-to-end pipeline: data -> temporal graph -> communities -> motifs -> features -> models -> alerts."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import alerts as al
from . import plots
from .community import community_report, community_risk, detect_communities, instance_cohesion
from .config import ALERT_DIR, DAY, FIG_DIR, RESULTS_DIR, TYPOLOGIES, DetectorConfig, ModelConfig, SimConfig
from .data import acquire
from .features import neighbour_features, structural_features, tabular_features, temporal_flow_features
from .graph import build_igraph, collapse_edges, summarise, undirected_weighted, weekly_snapshots
from .model import bootstrap_pr_auc_diff, evaluate, fit_model, grouped_split, per_typology_recall, shap_contributions
from .motifs import detect_all, evaluate_detectors, motif_node_features

FEATURE_SETS = ("tabular", "tabular+structural", "tabular+structural+temporal", "graph (all)")


def build_features(
    bank, det, links, mat, g, gu, membership, dcfg: DetectorConfig, mcfg: ModelConfig
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    acc, tx = bank.accounts, bank.transactions
    horizon = float(tx["ts"].max())
    tab = tabular_features(acc, tx, horizon)
    mot = motif_node_features(len(acc), det, links, mat, dcfg)
    st = structural_features(g, gu, membership, mot["motif_any"].to_numpy(), mcfg.betweenness_sources, mcfg.seed)
    tf = temporal_flow_features(tab, tx)
    tfm = pd.concat([tf, mot[["flow_links_through", "flow_median_hold_h", "flow_out_links", "max_in_72h", "max_out_72h"]]], axis=1)
    motif_cols = [c for c in mot.columns if c.startswith("motif_")]
    base = pd.concat([tab, st, tfm, mot[motif_cols]], axis=1)
    nbr = neighbour_features(gu, base, ["flow_through_ratio", "flow_links_through", "pagerank", "account_age_days", "motif_any"])
    x = pd.concat([base, nbr], axis=1)
    sets = {
        "tabular": list(tab.columns),
        "tabular+structural": list(tab.columns) + list(st.columns),
        "tabular+structural+temporal": list(tab.columns) + list(st.columns) + list(tfm.columns),
        "graph (all)": list(x.columns),
    }
    return x, sets


def run(
    sim: SimConfig = SimConfig(),
    dcfg: DetectorConfig = DetectorConfig(),
    mcfg: ModelConfig = ModelConfig(),
    out_dir: Path = RESULTS_DIR,
    try_public: bool = True,
    n_alerts: int | None = None,
) -> dict:
    t_all = time.time()
    timings: dict[str, float] = {}
    out_dir = Path(out_dir)
    fig_dir = out_dir / FIG_DIR.name
    alert_dir = out_dir / ALERT_DIR.name
    for d in (out_dir, fig_dir, alert_dir):
        d.mkdir(parents=True, exist_ok=True)
    for old in alert_dir.glob("*.html"):
        old.unlink()

    def tick(name: str, t0: float) -> float:
        timings[name] = round(time.time() - t0, 2)
        print(f"[{name}] {timings[name]:.1f}s", flush=True)
        return time.time()

    t = time.time()
    bundle = acquire(sim, try_public=try_public)
    bank = bundle.bank
    acc, tx = bank.accounts, bank.transactions
    y = acc["is_laundering"].to_numpy().astype(int)
    t = tick("data", t)

    pairs = collapse_edges(tx)
    g = build_igraph(len(acc), pairs)
    gu = undirected_weighted(len(acc), pairs)
    summary = summarise(tx, pairs, g)
    snap = weekly_snapshots(tx)
    snap.to_csv(out_dir / "weekly_snapshots.csv", index=False)
    t = tick("graph", t)

    memberships = detect_communities(gu, mcfg.seed)
    comm = community_report(gu, memberships, acc["region"].to_numpy())
    t = tick("communities", t)

    det, links, mat = detect_all(tx, dcfg)
    det_report, per_inst = evaluate_detectors(det, bank.instances, y) if len(bank.instances) else ({}, pd.DataFrame())
    if len(per_inst):
        per_inst.to_csv(out_dir / "typology_instances.csv", index=False)
    t = tick("motifs", t)

    membership = memberships["leiden"]
    x, sets = build_features(bank, det, links, mat, g, gu, membership, dcfg, mcfg)
    flagged = x["motif_any"].to_numpy() > 0
    comm["leiden_risk_ranking"] = community_risk(membership, flagged, y)
    if len(bank.instances):
        comm["leiden_instances_mostly_in_one_community"] = instance_cohesion(membership, bank.instances)
    ooc = x["out_of_community_share"].to_numpy()
    comm["out_of_community_share_mean"] = {"laundering": float(ooc[y == 1].mean()), "normal": float(ooc[y == 0].mean())}
    t = tick("features", t)

    tr, te = grouped_split(acc, mcfg)
    y_te = y[te]
    fits, evals = {}, {}
    for name in FEATURE_SETS:
        fits[name] = fit_model(name, x[sets[name]], y, tr, te, mcfg)
        evals[name] = evaluate(y_te, fits[name].test_score, mcfg)
        evals[name]["n_features"] = len(sets[name])
    s_tab, s_graph = fits["tabular"].test_score, fits["graph (all)"].test_score
    boot = bootstrap_pr_auc_diff(y_te, s_graph, s_tab, mcfg.n_bootstrap, mcfg.seed)
    t = tick("models", t)

    key = f"at_recall_{int(mcfg.target_recall * 100)}"
    acc_te = acc.iloc[te].reset_index(drop=True)
    model_typ = {
        name: per_typology_recall(acc_te, fits[name].test_score, evals[name][key]["threshold"]) for name in ("tabular", "graph (all)")
    }
    x_te = x.iloc[te]
    contrib = shap_contributions(fits["graph (all)"], x_te)
    imp = contrib.abs().mean().sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(out_dir / "feature_importance.csv")

    # ---------- alerts: top-N test accounts by graph-model score, each with an explainable subgraph
    n_alerts = n_alerts or mcfg.n_alerts
    order = np.argsort(-s_graph)[:n_alerts]
    score_map = al.score_lookup(acc_te["account_id"].to_numpy(), s_graph)
    truth = {int(a): t_ for a, t_ in zip(acc["account_id"], acc["typology"], strict=True) if t_ != "none"}
    flat = [d for ds in det.values() for d in ds]
    alert_rows = []
    first_png = None
    for rank, i in enumerate(order, start=1):
        a = int(acc_te["account_id"].iloc[i])
        sg = al.alert_subgraph(a, mat, links, flat)
        reasons = al.reason_codes(contrib.iloc[i], x_te.iloc[i])
        fired = al.fired_detectors(a, det)
        fname = f"alert_{rank:02d}_acct_{a}.html"
        al.export_html(sg, alert_dir / fname, f"Alert #{rank}: account {a} (score {s_graph[i]:.3f})", reasons, fired, score_map, truth)
        if first_png is None and sg.number_of_edges() > 2:
            plots.subgraph_png(
                sg, fig_dir / "alert_subgraph_example.png", f"Alert #{rank}: account {a}, red = motif / flow-link transfers (SYNTHETIC)"
            )
            first_png = rank
        alert_rows.append(
            {
                "rank": rank,
                "account_id": a,
                "score": round(float(s_graph[i]), 4),
                "ground_truth_synthetic": acc_te["typology"].iloc[i],
                "detectors_fired": ";".join(fired),
                "subgraph_nodes": sg.number_of_nodes(),
                "subgraph_edges": sg.number_of_edges(),
                "motif_edges": sum(1 for *_, d in sg.edges(data=True) if d["motif"]),
                "top_reason": reasons[0]["feature"] if reasons else "",
                "html": f"alerts/{fname}",
            }
        )
    # one detector example per typology (true-positive structure), for the investigator gallery
    examples = []
    lab_set = set(acc.loc[acc["is_laundering"], "account_id"].astype(int))
    for typ in TYPOLOGIES:
        cand = [d for d in det.get(typ, []) if d.hub is not None and d.accounts <= lab_set]
        if not cand:
            continue
        d = cand[0]
        sub = al.alert_subgraph(int(d.hub), mat, links, [d], max_edges=40)
        fname = f"typology_{typ}_acct_{d.hub}.html"
        al.export_html(sub, alert_dir / fname, f"Typology example: {typ} hubbed at account {d.hub}", [], [typ], score_map, truth)
        examples.append(
            {"typology": typ, "hub_account": int(d.hub), "accounts": len(d.accounts), "transfers": len(d.tx_ids), "html": f"alerts/{fname}"}
        )
    alerts_df = pd.DataFrame(alert_rows)
    alerts_df.to_csv(out_dir / "alerts.csv", index=False)
    t = tick("alerts", t)

    # ---------- figures
    plots.degree_ccdf(np.array(g.indegree()), np.array(g.outdegree()), fig_dir / "degree_ccdf.png")
    plots.weekly(snap, fig_dir / "weekly_activity.png")
    plots.pr_curves(y_te, {k: f.test_score for k, f in fits.items()}, {k: evals[k]["pr_auc"] for k in fits}, fig_dir / "pr_curves.png")
    if det_report:
        plots.typology_recall({k: det_report[k]["recall"] for k in TYPOLOGIES}, model_typ["graph (all)"], fig_dir / "typology_recall.png")
    plots.importance(imp, fig_dir / "feature_importance.png")
    wl = pd.DataFrame([{"feature_set": k, "false_positives": evals[k][key]["false_positives"]} for k in FEATURE_SETS])
    plots.workload(wl, fig_dir / "alert_workload.png")
    t = tick("figures", t)

    timings["total"] = round(time.time() - t_all, 2)
    tab_fp = evals["tabular"][key]["false_positives"]
    gr_fp = evals["graph (all)"][key]["false_positives"]
    metrics = {
        "data": {
            "source": bundle.source,
            "note": bundle.note,
            "synthetic": bundle.source == "synthetic",
            "seed": sim.seed,
            "n_days": sim.n_days,
            "laundering_accounts": int(y.sum()),
            "laundering_account_rate": float(y.mean()),
            "laundering_transactions": int(tx["is_laundering"].sum()),
            "typology_instances": int(bank.instances["instance_id"].nunique()) if len(bank.instances) else 0,
            "slow_instance_share": float(per_inst["slow"].mean()) if len(per_inst) else None,
        },
        "graph": summary,
        "communities": comm,
        "detectors": {"thresholds": dcfg.__dict__, "flow_links": int(len(links)), "material_transfers": int(len(mat)), **det_report},
        "models": {
            "split": {
                "train_accounts": int(len(tr)),
                "test_accounts": int(len(te)),
                "test_positives": int(y_te.sum()),
                "grouped_by": "laundering scheme instance",
            },
            "feature_sets": evals,
            "graph_minus_tabular_pr_auc_bootstrap": boot,
            "per_typology_account_recall_at_80": model_typ,
            "false_alert_reduction_at_80_recall": float(1 - gr_fp / tab_fp) if tab_fp else None,
            "top_features": imp.head(10).round(4).to_dict(),
        },
        "alerts": {
            "n_alerts": len(alert_rows),
            "precision_top_alerts": float((alerts_df["ground_truth_synthetic"] != "none").mean()),
            "alerts_with_motif_subgraph": int((alerts_df["motif_edges"] > 0).sum()),
            "typology_examples": examples,
        },
        "acceptance": {
            "each_typology_recall_reported": bool(det_report) and all(k in det_report for k in TYPOLOGIES),
            "graph_beats_tabular_pr_auc": bool(evals["graph (all)"]["pr_auc"] > evals["tabular"]["pr_auc"] and boot["ci95_lo"] > 0),
            "alerts_have_explainable_subgraph": bool(len(alert_rows) and (alerts_df["subgraph_edges"] > 0).all()),
        },
        "runtime_seconds": timings,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(metrics, alerts_df, out_dir / "report.md")
    return metrics


def write_report(m: dict, alerts_df: pd.DataFrame, path: Path) -> None:
    key = "at_recall_80"
    lines = ["# Day 13 results report (SYNTHETIC data)", "", f"Data: {m['data']['note']}", "", "## Graph", ""]
    lines += [f"- {k}: {v}" for k, v in m["graph"].items()]
    lines += [
        "",
        "## Typology detectors (instance recall, label-free rules)",
        "",
        "| Typology | Instances | Recall | Recall fast | Recall slow | Flagged accounts | Account precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TYPOLOGIES:
        d = m["detectors"].get(t)
        if d:
            lines.append(
                f"| {t} | {d['instances']} | {d['recall']:.3f} | {d['recall_fast']:.3f} | {d['recall_slow']:.3f} | {d['flagged_accounts']:,} | {d['account_precision']:.3f} |"
            )
    lines += [
        "",
        "## Models (held-out accounts, grouped by scheme)",
        "",
        "| Feature set | Features | PR-AUC | ROC-AUC | Precision@top1% | Alerts @80% recall | False alerts @80% recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, e in m["models"]["feature_sets"].items():
        o = e[key]
        lines.append(
            f"| {k} | {e['n_features']} | {e['pr_auc']:.3f} | {e['roc_auc']:.3f} | {e['precision_at_top1pct']:.3f} | {o['alerts']:,} | {o['false_positives']:,} |"
        )
    b = m["models"]["graph_minus_tabular_pr_auc_bootstrap"]
    lines += [
        "",
        f"Paired bootstrap PR-AUC(graph) - PR-AUC(tabular): {b['mean']:.3f} (95% CI {b['ci95_lo']:.3f} to {b['ci95_hi']:.3f}, n={b['n']})",
        "",
    ]
    lines += ["## Top alerts", "", md_table(alerts_df)]
    path.write_text("\n".join(lines) + "\n")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


__all__ = ["run", "build_features", "FEATURE_SETS", "DAY", "md_table"]
