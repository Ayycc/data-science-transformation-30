"""Markdown report generated from metrics (every number comes from the run)."""

from __future__ import annotations

from pathlib import Path


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _ci(d: dict, key: str = "t_ci", pct: bool = True) -> str:
    lo, hi = d[key]
    return f"[{lo * 100:+.2f}%, {hi * 100:+.2f}%]" if pct else f"[{lo:.2f}, {hi:.2f}]"


def policy_table(P: dict) -> str:
    rows = [
        "| Policy | Uplift vs static [95% t-CI] | Bootstrap CI | Seeds beating static | RevPAR $ | Occupancy | ADR $ | Cum. regret $k | Regret exponent | Regret $/night by year | Volatility | Max nightly change | Guardrail violations |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, v in P.items():
        u = v["uplift_vs_static"]
        reg = "n/a" if n == "oracle" else f"{v['cum_regret_final'] / 1e3:,.0f}"
        exp = "n/a" if n == "oracle" else f"{v['regret_exponent_mean_curve']:.2f}"
        rpn = "n/a" if n == "oracle" else " -> ".join(f"{x:.0f}" for x in v["regret_per_night_by_year"])
        rows.append(
            f"| {n} | {_pct(u['mean'])} {_ci(u)} | {_ci(u, 'boot_ci')} | {v['seeds_beating_static']}/{u['n']} | {v['revpar']['mean']:.2f} | {v['occupancy']:.1%} | {v['adr']:.2f} | {reg} | {exp} | {rpn} | {v['volatility_mean_abs_dlog']:.1%} | {v['max_rel_change']:.1%} | {v['guardrail_violations_posted']} |"
        )
    return "\n".join(rows)


def write(m: dict, path: Path) -> None:
    E = m["elasticity"]
    S = m["study"]
    cfgs = S["config"]
    g = cfgs["guard"]
    lines = [
        "# Day 27 report: dynamic pricing with contextual bandits (SYNTHETIC)",
        "",
        f"Data path: `{m['data']['path_used']}`. {m['data']['note']}",
        "",
        "## Logged year and elasticity estimation",
        "",
        f"{m['logs']['rows']} (night, booking window) cells over {m['logs']['nights']} nights; {m['logs']['randomised_test_cells']} randomised price-test cells. "
        f"Occupancy {m['logs']['occupancy']:.1%}, sell-out nights {m['logs']['sellout_share']:.1%}. "
        f"On manager-priced cells, corr(log price multiplier, log demand) = {m['logs']['corr_logprice_vs_logdemand_rm_cells']:+.3f} (price follows demand).",
        "",
        "| Estimator | Semi-elasticity theta | 95% CI | Covers true mean | Rows |",
        "|---|---|---|---|---|",
    ]
    for k in ("naive_ols", "randomised_test_ols", "iv_2sls"):
        lines.append(f"| {k} | {E[k]['theta']:.3f} | {_ci(E[k], 'ci95', pct=False)} | {E[k]['covers_truth_mean']} | {E[k]['n']} |")
    lines += [
        f"| truth (simulator, mean over cells at the static rate) | {E['truth']['theta_mean']:.3f} | p10-p90 [{E['truth']['theta_p10']:.2f}, {E['truth']['theta_p90']:.2f}] | | |",
        "",
        f"2SLS first-stage F = {E['iv_2sls']['first_stage_F']:.0f}.",
        "",
        "## Policy study",
        "",
        f"{cfgs['n_seeds']} seeds x {cfgs['n_nights']} nights x 3 booking windows; guardrails floor ${g['floor']:.0f}, ceiling ${g['ceiling']:.0f}, max {g['max_change']:.0%} change per night per window. "
        f"Bandits warm-started from randomised cells only: {cfgs['warm_start']}.",
        "",
        policy_table(S["policies"]),
        "",
        "Regret exponent = slope of log mean cumulative regret on log night over the last 75% of the horizon (1 = linear).",
        "",
        f"## Stretch: competitor reaction (strength {m['competitor_reaction']['strength']}, {m['competitor_reaction']['seeds']} seeds)",
        "",
        policy_table(m["competitor_reaction"]["policies"]),
        "",
        f"## Stretch: off-policy evaluation from {m['ope']['_n_cells']} randomised log cells",
        "",
        "| Target policy | Truth $/cell | IPS [95% CI] | SNIPS | DM | DR [95% CI] | IPS rel. err | DM rel. err | DR rel. err | Matched cells |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in m["ope"].items():
        if name.startswith("_"):
            continue
        lines.append(
            f"| {name} | {r['truth']:.0f} | {r['ips']:.0f} [{r['ips_ci'][0]:.0f}, {r['ips_ci'][1]:.0f}] | {r['snips']:.0f} | {r['dm']:.0f} | {r['dr']:.0f} [{r['dr_ci'][0]:.0f}, {r['dr_ci'][1]:.0f}] | {r['ips_rel_error']:+.1%} | {r['dm_rel_error']:+.1%} | {r['dr_rel_error']:+.1%} | {r['matched_cells']} |"
        )
    lines += ["", f"Run time: {m['timings']['total_seconds']:.0f} s.", ""]
    path.write_text("\n".join(lines))
