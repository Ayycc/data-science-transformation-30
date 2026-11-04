"""End-to-end run: data -> DP tables + budget -> query server -> DP model -> synthesisers -> utility/risk -> memo."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .accounting import BudgetExceededError, PrivacyAccountant, zcdp_to_dp
from .data import COLUMNS, PopulationConfig, generate_population, load_dataset, split
from .dp_model import run_dp_models
from .evaluate import anonymeter_risks, dcr_membership_inference, marginal_fidelity, qi_uniqueness, rare_subgroup_fidelity, tstr
from .query_server import QueryServer
from .release import ReleasePlan, accuracy_curves, composition_comparison, official_release
from .sdc import generalise, k_anonymise, k_anonymity_report
from .synth.copula import GaussianCopula
from .synth.ctgan_np import CTGANLite
from .synth.mst import DPTreeSynthesizer


@dataclass
class RunConfig:
    data_mode: str = "auto"
    population: PopulationConfig = field(default_factory=PopulationConfig)
    seed: int = 20261104
    # production privacy budget of the statistics office for this file
    epsilon_total: float = 4.0
    delta_total: float = 1e-5
    plan: ReleasePlan = field(default_factory=ReleasePlan)
    query_server_epsilon: float = 0.5
    query_epsilon_each: float = 0.05
    dp_model_epsilon: float = 1.0
    dp_synth_epsilon: float = 1.0
    dp_synth_delta: float = 1e-6
    # research sweeps (what-if analyses; see README on how they relate to the budget)
    curve_epsilons: list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    curve_trials: int = 300
    model_epsilons: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.3, 1.0, 3.0, 10.0])
    model_seeds: int = 10
    synth_epsilons: list[float] = field(default_factory=lambda: [0.3, 1.0, 3.0, 10.0])
    ctgan_epochs: int = 150
    n_attacks: int = 500
    mia_n: int = 2000
    k: int = 5
    # memo thresholds (set before looking at results)
    max_attack_risk: float = 0.05
    max_mia_auc: float = 0.55
    utility_tolerance: float = 0.005


def small_config() -> RunConfig:
    return RunConfig(
        population=PopulationConfig(n=3000),
        curve_trials=20,
        curve_epsilons=[0.1, 1.0],
        model_epsilons=[1.0],
        model_seeds=2,
        synth_epsilons=[1.0],
        ctgan_epochs=2,
        n_attacks=100,
        mia_n=300,
    )


def _query_script(cfg: RunConfig) -> list[tuple]:
    """A fixed analyst session: distinct count/mean queries plus repeats (which must be served from cache)."""
    q: list[tuple] = []
    for race in ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]:
        q.append(("count", {"race": race, "income": ">50K"}))
        q.append(("mean", "hours_per_week", {"race": race}))
    q += [("count", {"race": "White", "income": ">50K"}), ("mean", "hours_per_week", {"race": "Amer-Indian-Eskimo"})] * 3  # repeats
    for sex in ["Male", "Female"]:
        q.append(("mean", "age", {"sex": sex}))
        q.append(("count", {"sex": sex, "income": ">50K"}))
    return q


def run_query_server(df: pd.DataFrame, acc: PrivacyAccountant, cfg: RunConfig) -> dict:
    server = QueryServer(df, acc.sub_budget("query server", cfg.query_server_epsilon), seed=cfg.seed + 11)
    rows = []
    for item in _query_script(cfg):
        if item[0] == "count":
            ans = server.count(item[1], cfg.query_epsilon_each)
            m = np.ones(len(df), dtype=bool)
            for c, v in item[1].items():
                m &= (df[c] == v).to_numpy()
            truth = float(m.sum())
        else:
            ans = server.mean(item[1], item[2], cfg.query_epsilon_each)
            m = np.ones(len(df), dtype=bool)
            for c, v in item[2].items():
                m &= (df[c] == v).to_numpy()
            truth = float(df.loc[m, item[1]].mean())
        rows.append(
            {"query": ans.query, "answer": ans.value, "truth": truth, "cached": ans.cached, "refused": ans.refused, "eps": ans.epsilon_charged}
        )
    s = server.summary()
    assert server.accountant.spent_epsilon <= cfg.query_server_epsilon + 1e-12
    return {"summary": s, "log": rows}


def evaluate_release(name: str, release: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame, cfg: RunConfig, transform=None, **extra) -> dict:
    holdout_eval = transform(holdout) if transform else holdout
    train_eval = transform(train) if transform else train
    return {
        "name": name,
        "rows": int(len(release)),
        **extra,
        "fidelity": marginal_fidelity(train_eval, release),
        "tstr": tstr(release, holdout_eval, cfg.seed),
        "rare_subgroup": rare_subgroup_fidelity(train_eval, release),
        "mia_dcr": dcr_membership_inference(train, holdout, release, cfg.mia_n, cfg.seed, transform=transform),
        "qi_uniqueness": qi_uniqueness(train_eval, holdout_eval, release),
        "anonymeter": anonymeter_risks(train_eval, holdout_eval, release, cfg.n_attacks, cfg.seed),
    }


def recommend(releases: list[dict], cfg: RunConfig) -> dict:
    ok = []
    for r in releases:
        if r["name"].startswith("Raw"):
            continue
        a = r["anonymeter"]
        passes = (
            all(a[k]["ci"][0] <= cfg.max_attack_risk for k in ("singling_out", "linkability", "inference_capital_gain"))
            and r["mia_dcr"]["auc"] <= cfg.max_mia_auc
        )
        r["passes_risk_gate"] = bool(passes)
        if passes:
            ok.append(r)
    ok.sort(key=lambda r: (r["tstr"]["auc"], r.get("formal_dp", False)), reverse=True)
    best = ok[0] if ok else None
    # among DP files: the smallest epsilon whose utility is within `utility_tolerance` AUC of the best DP file
    dps = [r for r in ok if r.get("formal_dp")]
    best_dp = None
    if dps:
        top = max(r["tstr"]["auc"] for r in dps)
        best_dp = min((r for r in dps if r["tstr"]["auc"] >= top - cfg.utility_tolerance), key=lambda r: r["epsilon"])
    return {
        "passing": [r["name"] for r in ok],
        "best": best["name"] if best else None,
        "best_formal_dp": best_dp["name"] if best_dp else None,
        "rule": "pass = every Anonymeter risk has 95% CI lower bound <= max_attack_risk and DCR MIA AUC <= max_mia_auc; "
        "DP pick = smallest epsilon within utility_tolerance TSTR AUC of the best passing DP file",
    }


def run(cfg: RunConfig, out_dir: Path, data_dir: Path, sample_dir: Path | None = None, log=print) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def tick(name: str) -> None:
        timings[name] = round(time.perf_counter() - t0 - sum(timings.values()), 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    ds = load_dataset(cfg.data_mode, cfg.population)
    df = ds.df
    log(f"data path: {ds.path_used} ({len(df)} rows); attempts: {ds.attempts}")
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        tag = "SYNTHETIC" if ds.path_used == "synthetic_generator" else "UCI_ADULT"
        df.head(3000).to_csv(sample_dir / f"census_{tag}_sample.csv", index=False)
    train, holdout = split(df, cfg.seed)
    pilot = split(generate_population(PopulationConfig(n=min(len(df), 20000), seed=cfg.population.seed + 1)), cfg.seed + 1)
    tick("data")

    acc = PrivacyAccountant(cfg.epsilon_total, cfg.delta_total)
    res: dict = {
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "data": {
            "path_used": ds.path_used,
            "attempts": ds.attempts,
            "rows": len(df),
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "pilot_rows": len(pilot[0]),
        },
        "k_anonymity_raw": k_anonymity_report(train, k=cfg.k),
    }

    # 1. DP aggregate tables
    res["aggregate_release"] = official_release(train, acc, cfg.plan, cfg.seed)
    res["accuracy_curves"] = accuracy_curves(train, cfg.curve_epsilons, cfg.curve_trials, cfg.seed)
    res["composition"] = composition_comparison(1.0, 1e-6, [1, 2, 5, 10, 20, 50, 100, 200, 500])
    tick("aggregates")
    log(f"aggregate tables released; eps spent so far {acc.spent_epsilon:.3f}/{acc.epsilon_total}")

    # 2. query server
    res["query_server"] = run_query_server(train, acc, cfg)
    tick("query_server")
    log(f"query server: {res['query_server']['summary']['answered_new']} answered, {res['query_server']['summary']['refused']} refused")

    # 3. DP model
    acc.spend(f"DP logistic regression (objective perturbation) eps={cfg.dp_model_epsilon}", cfg.dp_model_epsilon)
    res["dp_model"] = run_dp_models(train, holdout, pilot, cfg.model_epsilons, cfg.model_seeds, cfg.seed)
    tick("dp_model")
    log(f"DP model done; baseline AUC {res['dp_model']['baseline']['auc']:.4f}")

    # 4. synthesisers + SDC
    releases = []
    releases.append(evaluate_release("Raw microdata (reference, not a release option)", train, train, holdout, cfg, formal_dp=False, epsilon=None))
    tick("eval_raw")
    kanon, kinfo = k_anonymise(train, cfg.k)
    releases.append(
        evaluate_release(
            f"k-anonymised microdata (k={cfg.k})", kanon, train, holdout, cfg, transform=generalise, formal_dp=False, epsilon=None, sdc=kinfo
        )
    )
    tick("eval_kanon")
    gc = GaussianCopula(seed=cfg.seed).fit(train)
    releases.append(evaluate_release(gc.name, gc.sample(len(train)), train, holdout, cfg, formal_dp=False, epsilon=None))
    tick("copula")
    log("copula evaluated")
    ct = CTGANLite(epochs=cfg.ctgan_epochs, seed=cfg.seed).fit(train, log=log)
    tick("ctgan_fit")
    releases.append(
        evaluate_release(
            ct.name,
            ct.sample(len(train)),
            train,
            holdout,
            cfg,
            formal_dp=False,
            epsilon=None,
            loss_history=[list(map(float, h)) for h in ct.history_[:: max(1, cfg.ctgan_epochs // 30)]],
        )
    )
    tick("ctgan_eval")
    log("CTGAN evaluated")
    dp_structures = {}
    for eps in cfg.synth_epsilons:
        m = DPTreeSynthesizer(eps, cfg.dp_synth_delta, seed=cfg.seed + int(eps * 100)).fit(train)
        assert m.epsilon_spent_ <= eps + 1e-9
        dp_structures[str(eps)] = m.structure()
        releases.append(
            evaluate_release(
                m.name,
                m.sample(len(train)),
                train,
                holdout,
                cfg,
                formal_dp=True,
                epsilon=eps,
                delta=cfg.dp_synth_delta,
                epsilon_spent=m.epsilon_spent_,
                rho_spent=m.accountant.spent,
            )
        )
        log(f"DP tree eps={eps} evaluated")
    tick("dp_synth")
    # the released DP synthetic file is charged to the production budget
    acc.spend(f"DP synthetic microdata (MST-style) eps={cfg.dp_synth_epsilon}", cfg.dp_synth_epsilon, cfg.dp_synth_delta)
    res["dp_synth_structures"] = dp_structures
    res["releases"] = releases
    res["recommendation"] = recommend(releases, cfg)

    # 5. budget enforcement demo: one more spend that does not fit must be refused
    try:
        acc.spend("over-budget request (should be refused)", acc.remaining_epsilon + 0.25)
        refused = False
    except BudgetExceededError:
        refused = True
    res["budget"] = acc.to_dict() | {
        "over_budget_request_refused": refused,
        "within_budget": acc.spent_epsilon <= acc.epsilon_total + 1e-12 and acc.spent_delta <= acc.delta_total,
    }
    rel = next((r for r in releases if r.get("epsilon") == cfg.dp_synth_epsilon), None)
    if rel is not None:
        res["zcdp_check"] = {
            "epsilon": cfg.dp_synth_epsilon,
            "rho_spent": rel["rho_spent"],
            "eps_from_rho": zcdp_to_dp(rel["rho_spent"], cfg.dp_synth_delta),
        }
    timings["total_seconds"] = round(time.perf_counter() - t0, 1)
    res["timings"] = timings

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=_json_default))
    from .figures import make_figures
    from .report import write_memo, write_report

    make_figures(res, out_dir / "figures")
    write_report(res, out_dir / "report.md")
    write_memo(res, out_dir / "release_memo.md")
    log(f"wrote {out_dir}")
    return res


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


__all__ = ["RunConfig", "small_config", "run", "COLUMNS"]
