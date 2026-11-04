"""Markdown report (every table) and the release recommendation memo, generated from the run's results."""

from __future__ import annotations

from pathlib import Path


def _f(x, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _pct(x, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _short(r: dict) -> str:
    return f"DP tree ε={r['epsilon']:g}" if r.get("formal_dp") else r["name"]


def tables(res: dict) -> dict[str, str]:
    t: dict[str, str] = {}
    b = res["budget"]
    lines = ["| Spend | ε | δ |", "|---|---|---|"]
    lines += [f"| {s['label']} | {s['epsilon']:g} | {s['delta']:g} |" for s in b["ledger"]]
    lines.append(f"| **Total spent / budget** | **{b['epsilon_spent']:g} / {b['epsilon_total']:g}** | {b['delta_spent']:g} / {b['delta_total']:g} |")
    t["budget"] = "\n".join(lines)

    ag = res["aggregate_release"]
    T = ag["tables"]
    lines = ["| Table | ε | Cells | Mean abs error | Max abs error | Median relative error (non-zero cells) |", "|---|---|---|---|---|---|"]
    eps = {s["label"].split(" ")[0]: s["epsilon"] for s in ag["budget"]["ledger"]}
    for key, lab in [("T1_race_sex_income", "T1"), ("T2_age_histogram", "T2"), ("T4_education_income", "T4")]:
        x = T[key]
        lines.append(
            f"| {lab} {key[3:].replace('_', ' ')} | {eps[lab]:g} | {x['cells']} | {_f(x['mean_abs_error'], 2)} | {_f(x['max_abs_error'], 1)} | {_pct(x['median_rel_error_nonzero_cells'], 2)} |"
        )
    t["aggregates"] = "\n".join(lines)
    t1 = T["T1_race_sex_income"]
    lines = ["| Amer-Indian-Eskimo cell (sex, income) | True count | Released (ε=" + f"{eps['T1']:g}) |", "|---|---|---|"]
    for k, v in t1["rare_group_cells"].items():
        lines.append(f"| {k} | {v['exact']:.0f} | {v['released']:.0f} |")
    t["rare_cells"] = "\n".join(lines)
    t3 = T["T3_mean_hours_by_race"]
    lines = ["| Race | True mean hours | DP mean hours | Abs error |", "|---|---|---|---|"]
    for k in t3["exact"]:
        lines.append(f"| {k} | {t3['exact'][k]:.2f} | {t3['dp'][k]:.2f} | {t3['abs_error'][k]:.2f} |")
    t["t3"] = "\n".join(lines)

    c = res["accuracy_curves"]
    lines = [
        "| ε | Count rel. error, White cells | Count rel. error, Amer-Indian-Eskimo cells | Ratio rare / White | Age histogram TVD | Mean hours abs error White / Amer-Indian-Eskimo |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(c["epsilon"]):
        w, r = c["count_rel_err_white"][i], c["count_rel_err_rare"][i]
        lines.append(
            f"| {e:g} | {_pct(w, 3)} | {_pct(r, 2)} | {r / w:.0f}x | {_f(c['age_hist_tvd'][i], 4)} | {_f(c['mean_hours_abs_err_white'][i], 3)} / {_f(c['mean_hours_abs_err_rare'][i], 2)} |"
        )
    t["curves"] = "\n".join(lines)

    comp = res["composition"]
    lines = ["| Queries k | Laplace + basic composition, noise std | Gaussian + zCDP, noise std |", "|---|---|---|"]
    for k, a, g in zip(comp["k"], comp["laplace_std"], comp["gaussian_zcdp_std"]):
        lines.append(f"| {k} | {a:.1f} | {g:.1f} |")
    t["composition"] = "\n".join(lines)

    qs = res["query_server"]
    lines = ["| # | Query | DP answer | True value | Charged ε | Status |", "|---|---|---|---|---|---|"]
    for i, q in enumerate(qs["log"], 1):
        status = "refused (budget)" if q["refused"] else ("cache hit" if q["cached"] else "answered")
        ans = "—" if q["answer"] is None else f"{q['answer']:.1f}"
        lines.append(f"| {i} | `{q['query']}` | {ans} | {q['truth']:.1f} | {q['eps']:g} | {status} |")
    t["query_server"] = "\n".join(lines)

    dm = res["dp_model"]
    bl = dm["baseline"]
    lines = [
        "| Model | ε | λ (chosen on pilot) | AUC | Accuracy | AUC Amer-Indian-Eskimo | Accuracy Amer-Indian-Eskimo | Loss-threshold MIA AUC |",
        "|---|---|---|---|---|---|---|---|",
        f"| Non-private logistic regression | ∞ | C=1 | {_f(bl['auc'], 4)} | {_f(bl['accuracy'], 4)} | {_f(bl['auc_rare'], 4)} | {_f(bl['accuracy_rare'], 4)} | {_f(bl['mia_auc'], 4)} |",
    ]
    for r in dm["dp"]:
        lines.append(
            f"| DP objective perturbation (mean of {r['seeds']}) | {r['epsilon']:g} | {r['lambda']:g} | {_f(r['auc'], 4)} ± {_f(r['auc_sd'], 4)} | {_f(r['accuracy'], 4)} | {_f(r['auc_rare'], 4)} ± {_f(r['auc_rare_sd'], 3)} | {_f(r['accuracy_rare'], 4)} | {_f(r['mia_auc'], 4)} |"
        )
    for r in dm["diffprivlib_check"]:
        lines.append(
            f"| diffprivlib 0.6.5 LogisticRegression (cross-check) | {r['epsilon']:g} | C=1 | {_f(r['auc'], 4)} | {_f(r['accuracy'], 4)} | {_f(r['auc_rare'], 4)} | {_f(r['accuracy_rare'], 4)} | |"
        )
    lines.append(f"| Always predict <=50K | | | 0.5 | {_f(dm['majority_class_accuracy'], 4)} | 0.5 | {_f(dm['majority_class_accuracy_rare'], 4)} | |")
    t["dp_model"] = "\n".join(lines)

    rels = res["releases"]
    lines = [
        "| Release | Formal DP | Rows | 1-way TVD | 2-way TVD (mean / max) | TSTR AUC | TSTR accuracy | TSTR AUC Amer-Indian-Eskimo | AUC gap rare vs other |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rels:
        f, s = r["fidelity"], r["tstr"]
        dp = f"ε={r['epsilon']:g}, δ={r['delta']:g}" if r.get("formal_dp") else "no"
        lines.append(
            f"| {_short(r)} | {dp} | {r['rows']:,} | {_f(f['tvd_1way_mean'], 4)} | {_f(f['tvd_2way_mean'], 4)} / {_f(f['tvd_2way_max'], 3)} | {_f(s['auc'], 4)} | {_f(s['accuracy'], 4)} | {_f(s['auc_rare'], 4)} | {s['auc_rare'] - s['auc_other']:+.4f} |"
        )
    t["utility"] = "\n".join(lines)

    lines = [
        "| Release | Singling-out risk [95% CI] | Linkability risk [95% CI] | Inference risk, capital gain [95% CI] | DCR membership AUC | Exact copies of members / non-members | Rows matching a train-unique QI combo (holdout baseline) | Passes gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rels:
        a, m, q = r["anonymeter"], r["mia_dcr"], r["qi_uniqueness"]

        def ci(k, a=a):
            return f"{_f(a[k]['risk'], 3)} [{_f(a[k]['ci'][0], 3)}, {_f(a[k]['ci'][1], 3)}]"

        gate = "reference" if r["name"].startswith("Raw") else ("yes" if r.get("passes_risk_gate") else "no")
        lines.append(
            f"| {_short(r)} | {ci('singling_out')} | {ci('linkability')} | {ci('inference_capital_gain')} | {_f(m['auc'], 4)} | {_pct(m['exact_match_rate_members'])} / {_pct(m['exact_match_rate_non_members'])} | {_pct(q['release_rows_matching_train_unique'])} ({_pct(q['holdout_rows_matching_train_unique'])}) | {gate} |"
        )
    t["risk"] = "\n".join(lines)

    lines = [
        "| Release | Rare share (real) | Rare share (release) | Rare >50K rate (real) | Rare >50K rate (release) | Within-group TVD rare / White |",
        "|---|---|---|---|---|---|",
    ]
    for r in rels:
        g = r["rare_subgroup"]
        lines.append(
            f"| {_short(r)} | {_pct(g['real_share'], 2)} | {_pct(g['release_share'], 2)} | {_pct(g['real_positive_rate'])} | {_pct(g['release_positive_rate'])} | {_f(g['conditional_tvd_rare'], 3)} / {_f(g['conditional_tvd_majority'], 3)} |"
        )
    t["rare"] = "\n".join(lines)
    return t


def write_report(res: dict, path: Path) -> None:
    t = tables(res)
    k = res["k_anonymity_raw"]
    kan = next(r for r in res["releases"] if r["name"].startswith("k-anon"))
    md = [
        "# Day 25 results (generated by `make run`)",
        "",
        f"Data path: **{res['data']['path_used']}** ({res['data']['rows']:,} rows; {res['data']['train_rows']:,} confidential / {res['data']['holdout_rows']:,} holdout). Runtime {res['timings']['total_seconds']:.0f} s.",
        "",
        "## Production privacy budget",
        "",
        t["budget"],
        "",
        f"Over-budget request refused: **{res['budget']['over_budget_request_refused']}**; within budget: **{res['budget']['within_budget']}**.",
        "",
        "## DP aggregate tables",
        "",
        t["aggregates"],
        "",
        t["rare_cells"],
        "",
        t["t3"],
        "",
        "## Accuracy vs epsilon (Monte Carlo)",
        "",
        t["curves"],
        "",
        "## Composition: Laplace/basic vs Gaussian/zCDP (ε=1, δ=1e-6)",
        "",
        t["composition"],
        "",
        "## Query server session",
        "",
        t["query_server"],
        "",
        "## DP model",
        "",
        t["dp_model"],
        "",
        "## Released microdata: utility",
        "",
        t["utility"],
        "",
        "## Released microdata: disclosure risk",
        "",
        t["risk"],
        "",
        f"k-anonymity of the raw confidential file on {', '.join(k['quasi_identifiers'])}: {k['equivalence_classes']:,} classes, "
        f"{_pct(k['share_records_unique'])} of records unique, {_pct(k['share_records_k_lt_5'])} in classes below k=5. "
        f"The k=5 release suppressed {kan['sdc']['suppressed_records']:,} records ({_pct(kan['sdc']['suppressed_share'])}).",
        "",
        "## Rare subgroup (Amer-Indian-Eskimo)",
        "",
        t["rare"],
        "",
        "## DP tree structures",
        "",
    ]
    for e, s in res["dp_synth_structures"].items():
        md.append(f"- ε={e}: " + "; ".join(s))
    md += ["", f"Recommendation: {res['recommendation']}", ""]
    path.write_text("\n".join(md))


def write_memo(res: dict, path: Path) -> None:
    rels = {r["name"]: r for r in res["releases"]}
    rec = res["recommendation"]
    cfg = res["config"]
    dm = res["dp_model"]
    raw = next(r for r in res["releases"] if r["name"].startswith("Raw"))
    best = rels.get(rec["best"]) if rec["best"] else None
    best_dp = rels.get(rec["best_formal_dp"]) if rec["best_formal_dp"] else None
    prod = next((r for r in res["releases"] if r.get("formal_dp") and r["epsilon"] == cfg["dp_synth_epsilon"]), None)
    m1 = next((r for r in dm["dp"] if r["epsilon"] == cfg["dp_model_epsilon"]), None)
    t1 = res["aggregate_release"]["tables"]["T1_race_sex_income"]
    lines = [
        "# Release recommendation memo (generated from results/metrics.json)",
        "",
        f"**Data:** {res['data']['path_used']} ({'SYNTHETIC census-like population' if res['data']['path_used'] == 'synthetic_generator' else 'UCI Adult'}), {res['data']['train_rows']:,} confidential records.",
        f"**Budget:** ε = {res['budget']['epsilon_total']:g}, δ = {res['budget']['delta_total']:g} for this file; spent ε = {res['budget']['epsilon_spent']:g}, δ = {res['budget']['delta_spent']:g}.",
        f"**Risk gate:** no Anonymeter risk significantly above {cfg['max_attack_risk']} (95% CI lower bound ≤ {cfg['max_attack_risk']}) and DCR membership-inference AUC ≤ {cfg['max_mia_auc']}. "
        f"Among DP files that pass, pick the smallest ε whose TSTR AUC is within {cfg['utility_tolerance']} of the best DP file.",
        "",
        "## Recommendation",
        "",
    ]
    lines.append(
        f"1. **Publish the four aggregate tables with Laplace noise (ε = {cfg['plan']['epsilon']:g} in total).** The median relative error of the "
        f"race × sex × income counts is {_pct(t1['median_rel_error_nonzero_cells'], 2)}, but Amer-Indian-Eskimo cells carry a mean relative error of "
        f"{_pct(t1['rare_group_mean_rel_error'])} against {_pct(t1['other_groups_mean_rel_error'])} for the other groups: publish those cells with a "
        "flag and their noise scale, and point users of the rare group to coarser tables."
    )
    if best_dp:
        lines.append(
            f"2. **Researcher microdata: release `{best_dp['name']}`** (the formally DP file picked by the rule above). TSTR AUC "
            f"{_f(best_dp['tstr']['auc'], 4)} vs {_f(raw['tstr']['auc'], 4)} on real data, max Anonymeter risk {_f(best_dp['anonymeter']['max_risk'], 3)}, "
            f"membership AUC {_f(best_dp['mia_dcr']['auc'], 4)}."
        )
        if prod and prod["name"] != best_dp["name"]:
            lines.append(
                f"   The production ledger reserved ε = {cfg['dp_synth_epsilon']:g} for this file; at that ε the TSTR AUC is {_f(prod['tstr']['auc'], 4)}. "
                f"Releasing ε = {best_dp['epsilon']:g} instead needs the budget to be raised (or other releases cut) by {best_dp['epsilon'] - cfg['dp_synth_epsilon']:+g}."
            )
    if best and best is not best_dp:
        lines.append(
            f"3. `{best['name']}` scores higher (TSTR AUC {_f(best['tstr']['auc'], 4)}) and passes the empirical gate, but it has **no formal guarantee**: "
            "its risk numbers hold only for the attacks we ran. Use it only inside a secure research environment, not for open release."
        )
    failing = [r for r in res["releases"] if not r["name"].startswith("Raw") and not r.get("passes_risk_gate")]
    if failing:
        lines.append(
            "4. **Do not release:** "
            + "; ".join(
                f"`{_short(r)}` (max Anonymeter risk {_f(r['anonymeter']['max_risk'], 3)}, membership AUC {_f(r['mia_dcr']['auc'], 3)})"
                for r in failing
            )
            + "."
        )
    if m1:
        lines.append(
            f"5. **Model release:** the ε = {cfg['dp_model_epsilon']:g} logistic regression reaches AUC {_f(m1['auc'], 4)} (non-private {_f(dm['baseline']['auc'], 4)}); "
            f"on the rare group AUC {_f(m1['auc_rare'], 4)} (non-private {_f(dm['baseline']['auc_rare'], 4)})."
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Empirical attack scores are lower bounds on risk; only the DP releases carry a worst-case guarantee.",
        f"- The rare subgroup has only {raw['tstr']['n_test_rare']} holdout records, so its AUCs have wide sampling error.",
        "- The ε sweeps (curves, model and synthesiser grids) are what-if analyses on the confidential file; in production only the ledger entries above would be run.",
    ]
    if res["data"]["path_used"] == "synthetic_generator":
        lines.append("- All numbers come from a SYNTHETIC population; the public UCI Adult file was unreachable from the build network.")
    path.write_text("\n".join(lines) + "\n")
