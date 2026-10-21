"""End-to-end run: data -> PII redaction -> normalisation -> baselines -> active learning ->
abstention -> topic discovery -> figures, metrics.json and report.md."""

from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import normalize

from . import plots
from .active import labels_to_match, schedule, simulate
from .config import FIG_DIR, RESULTS_DIR, SAMPLE_DIR, RunConfig
from .data import load_tickets
from .features import Embedder, make_tfidf
from .models import evaluate, full_proba, macro_f1, make_clf
from .pii import Redactor, evaluate_redaction, load_handlabelled
from .text import normalise, strip_boilerplate
from .topics import cluster, cluster_quality, cluster_table, umap_reduce

warnings.filterwarnings("ignore", message="The number of unique classes")
warnings.filterwarnings("ignore", category=FutureWarning)


def _r(x, nd: int = 4):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_r(v, nd) for v in x]
    if isinstance(x, float | np.floating):
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def abstention_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose auto-routed set reaches `target` accuracy (on held-out predictions)."""
    order = np.argsort(-conf)
    acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(acc >= target)
    if len(ok) == 0:
        return 1.01
    return float(conf[order][ok.max()])


def coverage_curve(conf: np.ndarray, correct: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-conf)
    acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    cov = np.arange(1, len(order) + 1) / len(order)
    idx = np.unique(np.linspace(0, len(order) - 1, n_points).astype(int))
    return cov[idx], acc[idx]


def run(cfg: RunConfig, results_dir: Path = RESULTS_DIR, source: str | None = None, write_sample: bool = True, n_jobs: int = 4) -> dict:
    t0 = time.time()
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_dir / "figures" if results_dir != RESULTS_DIR else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # 1. Data -------------------------------------------------------------------------------
    df, prov = load_tickets(cfg, source)
    on_synth = prov["path"] == "synthetic"
    holdout = list(cfg.holdout_intents_synthetic if on_synth else cfg.holdout_intents)
    labels = sorted(df["intent"].unique())
    _log(f"data: {prov['path']} ({len(df)} tickets, {len(labels)} intents)", t0)
    if write_sample and prov["path"] == "bitext":
        sample = df.groupby("intent", group_keys=False).head(60)
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        sample.to_csv(SAMPLE_DIR / "tickets_sample.csv", index=False)
    timings["data_s"] = time.time() - t0

    # 2. PII redaction ----------------------------------------------------------------------
    t = time.time()
    red = Redactor()
    texts = df["text"].tolist()
    redacted, spans = red.redact_many(texts)
    pii_main = evaluate_redaction(texts, df["pii_spans"].tolist(), spans)
    ablations = {}
    for name, kw in {"ner_plus_regex_no_cues": {"use_cues": False}, "regex_and_cues_no_ner": {"use_ner": False}}.items():
        p = Redactor(**kw).detect_many(texts)
        ablations[name] = evaluate_redaction(texts, df["pii_spans"].tolist(), p)
    hl_texts, hl_gold = load_handlabelled(SAMPLE_DIR / "pii_handlabelled.txt")
    pii_hand = evaluate_redaction(hl_texts, hl_gold, red.detect_many(hl_texts))
    df["redacted"] = redacted
    df["norm"] = [normalise(x) for x in redacted]
    df["core"] = [strip_boilerplate(x) for x in df["norm"]]
    leaked = int(sum(any(c in r for c in ("@example", "@mail.example", "@corp.example", "@inbox.test", "@acme-widgets")) for r in redacted))
    timings["pii_s"] = time.time() - t
    _log(f"pii: recall={pii_main['recall_any_label']:.4f} hand={pii_hand['recall_any_label']:.4f} ner={red.has_ner}", t0)

    # 3. Split & features -------------------------------------------------------------------
    t = time.time()
    pool, test = train_test_split(df, test_size=cfg.test_frac, stratify=df["intent"], random_state=cfg.seed)
    y_pool, y_test = pool["intent"].to_numpy(), test["intent"].to_numpy()
    tfidf = make_tfidf().fit(pool["norm"])  # unsupervised: unlabelled pool text is available
    X_pool, X_test = tfidf.transform(pool["norm"]), tfidf.transform(test["norm"])
    svd = TruncatedSVD(n_components=min(100, X_pool.shape[1] - 1), random_state=cfg.seed).fit(X_pool)
    Z_pool = normalize(svd.transform(X_pool))
    emb = Embedder()
    E_pool, E_test = emb.encode(pool["norm"].tolist()), emb.encode(test["norm"].tolist())
    timings["features_s"] = time.time() - t
    _log(f"features: tfidf {X_pool.shape}, embedder={emb.backend}", t0)

    # 4. Baselines (full labels and the 2% random budget) -----------------------------------
    t = time.time()
    init_n, batch_n, budget_n = schedule(len(y_pool), cfg.init_frac, cfg.batch_frac, cfg.budget_frac)
    baselines = {}
    full_models = {}
    for kind, A, B in (("tfidf", X_pool, X_test), ("emb", E_pool, E_test)):
        clf = make_clf(kind, cfg.seed).fit(A, y_pool)
        full_models[kind] = clf
        ev = evaluate(y_test, clf.predict(B), labels)
        baselines[f"{kind}_full_labels"] = ev
    timings["baselines_s"] = time.time() - t
    _log(f"baselines: tfidf={baselines['tfidf_full_labels']['macro_f1']:.4f} emb={baselines['emb_full_labels']['macro_f1']:.4f}", t0)

    # 5. Active learning --------------------------------------------------------------------
    t = time.time()
    jobs = []
    for rep in range(cfg.al_repeats):
        for s in ("random", "margin", "entropy", "margin_diverse"):
            jobs.append(("tfidf", s, rep))
        for s in ("random", "margin"):
            jobs.append(("emb", s, rep))
    feats = {"tfidf": (X_pool, X_test), "emb": (E_pool, E_test)}

    def _one(kind, s, rep):
        A, B = feats[kind]
        return simulate(A, y_pool, B, y_test, labels, kind, s, init_n, batch_n, budget_n, cfg.seed, rep, Z_pool=Z_pool)

    runs = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one)(*j) for j in jobs)
    timings["active_learning_s"] = time.time() - t
    curves_rows = []
    for r in runs:
        for n, f in zip(r.n_labels, r.macro_f1, strict=True):
            curves_rows.append({"features": r.kind, "strategy": r.strategy, "repeat": r.repeat, "n_labels": n, "macro_f1": f})
    curves = pd.DataFrame(curves_rows)
    curves.to_csv(results_dir / "al_curves.csv", index=False)
    agg = curves.groupby(["features", "strategy", "n_labels"])["macro_f1"].agg(["mean", "std"]).reset_index()

    al = {
        "pool_size": int(len(y_pool)),
        "test_size": int(len(y_test)),
        "init_labels": init_n,
        "batch_labels": batch_n,
        "budget_labels": budget_n,
        "repeats": cfg.al_repeats,
        "by_features": {},
    }
    for kind in ("tfidf", "emb"):
        a = agg[agg.features == kind]
        rnd = a[a.strategy == "random"]
        target = float(rnd[rnd.n_labels == budget_n]["mean"].iloc[0])
        res = {"random_f1_at_budget": target, "strategies": {}}
        for s in sorted(a.strategy.unique()):
            sa = a[a.strategy == s]
            need = labels_to_match(sa["mean"].tolist(), sa["n_labels"].tolist(), target)
            per_rep = []
            for rep in range(cfg.al_repeats):
                cr = curves[(curves.features == kind) & (curves.strategy == s) & (curves.repeat == rep)]
                rr = curves[(curves.features == kind) & (curves.strategy == "random") & (curves.repeat == rep)]
                tgt = float(rr["macro_f1"].iloc[-1])
                n_rep = labels_to_match(cr["macro_f1"].tolist(), cr["n_labels"].tolist(), tgt)
                per_rep.append(None if n_rep is None else n_rep / budget_n)
            res["strategies"][s] = {
                "f1_at_budget_mean": float(sa[sa.n_labels == budget_n]["mean"].iloc[0]),
                "f1_at_budget_std": float(sa[sa.n_labels == budget_n]["std"].iloc[0]),
                "f1_at_half_budget_mean": float(np.interp(budget_n / 2, sa["n_labels"], sa["mean"])),
                "labels_to_match_random_at_budget": need,
                "label_share_to_match_random": None if need is None else need / budget_n,
                "label_share_per_repeat": per_rep,
            }
        al["by_features"][kind] = res
        plots.learning_curves(
            {s: (g["n_labels"].tolist(), g["mean"].to_numpy(), g["std"].fillna(0).to_numpy()) for s, g in a.groupby("strategy")},
            len(y_pool),
            f"Active learning vs random, {'TF-IDF' if kind == 'tfidf' else 'embedding'} + logistic regression (mean ± sd, {cfg.al_repeats} repeats)",
            fig_dir / f"learning_curve_{kind}.png",
            target,
        )
    primary = al["by_features"]["tfidf"]["strategies"]["margin"]
    al["primary"] = "tfidf / margin"
    al["criterion_label_share_le_0_5_met"] = bool(
        primary["label_share_to_match_random"] is not None and primary["label_share_to_match_random"] <= 0.5
    )

    # Reverse view: how many random labels are needed to reach the AL F1 at the 2% budget?
    multiples = [1, 1.5, 2, 3, 4, 6, 10]
    ext = []
    for rep in range(cfg.al_repeats):
        rng = np.random.default_rng(cfg.seed + 1000 * rep + 99)
        perm = rng.permutation(len(y_pool))
        for mlt in multiples:
            n = min(int(budget_n * mlt), len(y_pool))
            idx = perm[:n]
            clf = make_clf("tfidf", cfg.seed).fit(X_pool[idx], y_pool[idx])
            ext.append({"repeat": rep, "n_labels": n, "macro_f1": macro_f1(y_test, clf.predict(X_test), labels)})
    ext = pd.DataFrame(ext).groupby("n_labels")["macro_f1"].mean().reset_index()
    ext.to_csv(results_dir / "random_extended_curve.csv", index=False)
    al_final = primary["f1_at_budget_mean"]
    reach = ext[ext.macro_f1 >= al_final]
    al["random_labels_to_reach_al_f1"] = None if reach.empty else int(reach.n_labels.iloc[0])
    al["random_extended_curve"] = ext.to_dict(orient="list")
    _log(f"active learning: share={primary['label_share_to_match_random']} ({timings['active_learning_s']:.0f}s)", t0)

    # Per-intent evaluation of the 2%-budget AL model (repeat 0) and of random at 2% (repeat 0)
    r0 = next(r for r in runs if r.kind == "tfidf" and r.strategy == "margin" and r.repeat == 0)
    rr0 = next(r for r in runs if r.kind == "tfidf" and r.strategy == "random" and r.repeat == 0)
    al_clf = make_clf("tfidf", cfg.seed).fit(X_pool[r0.labelled_idx], y_pool[r0.labelled_idx])
    rnd_clf = make_clf("tfidf", cfg.seed).fit(X_pool[rr0.labelled_idx], y_pool[rr0.labelled_idx])
    baselines["tfidf_al_margin_2pct"] = evaluate(y_test, al_clf.predict(X_test), labels)
    baselines["tfidf_random_2pct"] = evaluate(y_test, rnd_clf.predict(X_test), labels)
    emb_r = next(r for r in runs if r.kind == "emb" and r.strategy == "random" and r.repeat == 0)
    emb_rnd = make_clf("emb", cfg.seed).fit(E_pool[emb_r.labelled_idx], y_pool[emb_r.labelled_idx])
    baselines["emb_random_2pct"] = evaluate(y_test, emb_rnd.predict(E_test), labels)

    # 6. Calibrated abstention (stretch) ----------------------------------------------------
    t = time.time()
    lab_idx = r0.labelled_idx
    cv = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    cv_proba = cross_val_predict(make_clf("tfidf", cfg.seed), X_pool[lab_idx], y_pool[lab_idx], cv=cv, method="predict_proba")
    cv_classes = np.unique(y_pool[lab_idx])
    cv_conf = cv_proba.max(axis=1)
    cv_correct = (cv_classes[cv_proba.argmax(axis=1)] == y_pool[lab_idx]).astype(float)
    tau = abstention_threshold(cv_conf, cv_correct, cfg.abstain_target_acc)
    p_test = full_proba(al_clf, X_test, labels)
    conf = p_test.max(axis=1)
    pred = np.asarray(labels)[p_test.argmax(axis=1)]
    correct = (pred == y_test).astype(float)
    auto = conf >= tau
    abst = {
        "model": "tfidf + LR trained on the 2% margin-AL labels (repeat 0)",
        "target_accuracy": cfg.abstain_target_acc,
        "threshold_from_5fold_cv_on_labelled_set": tau,
        "test_coverage": float(auto.mean()),
        "test_accuracy_auto_routed": float(correct[auto].mean()) if auto.any() else None,
        "test_accuracy_all": float(correct.mean()),
        "test_accuracy_sent_to_humans": float(correct[~auto].mean()) if (~auto).any() else None,
    }
    full_p = full_proba(full_models["tfidf"], X_test, labels)
    full_correct = (np.asarray(labels)[full_p.argmax(axis=1)] == y_test).astype(float)
    plots.abstention_curve(
        {
            "2% labels (margin AL)": coverage_curve(conf, correct),
            "2% labels (random)": coverage_curve(
                full_proba(rnd_clf, X_test, labels).max(axis=1), (rnd_clf.predict(X_test) == y_test).astype(float)
            ),
            "100% labels": coverage_curve(full_p.max(axis=1), full_correct),
        },
        {"2% labels (margin AL)": (abst["test_coverage"], abst["test_accuracy_auto_routed"] or 0.0)},
        fig_dir / "abstention.png",
    )
    timings["abstention_s"] = time.time() - t

    # 7. Topic discovery --------------------------------------------------------------------
    t = time.time()
    t_tfidf = make_tfidf().fit(pool["core"])
    t_svd = TruncatedSVD(n_components=min(100, len(test) - 1), random_state=cfg.seed).fit(t_tfidf.transform(pool["core"]))
    Zt = normalize(t_svd.transform(t_tfidf.transform(test["core"])))
    U = umap_reduce(Zt, 5, cfg.umap_neighbors, cfg.seed)
    cl = cluster(U, cfg.min_cluster_size)
    topics = {"representation": "LSA: TF-IDF(boilerplate-stripped text) -> TruncatedSVD(100) -> L2", "quality": cluster_quality(y_test, cl)}
    table = cluster_table(test["core"].tolist(), y_test, cl)
    pd.DataFrame([{**r, "keywords": ", ".join(r["keywords"])} for r in table]).to_csv(results_dir / "topic_clusters.csv", index=False)
    topics["clusters_purity_ge_0_8"] = int(sum(r["purity"] >= 0.8 for r in table))
    topics["intents_with_a_pure_cluster"] = int(len({r["majority_intent"] for r in table if r["purity"] >= 0.8}))
    # comparison: the dense embedder (spaCy vectors) as the clustering representation
    Ue = umap_reduce(emb.encode(test["core"].tolist()), 5, cfg.umap_neighbors, cfg.seed)
    topics["quality_embedder"] = {"representation": emb.backend, **cluster_quality(y_test, cluster(Ue, cfg.min_cluster_size))}

    # Emerging issues: classifier never saw `holdout`; low-confidence tickets are clustered
    keep = ~np.isin(y_pool, holdout)
    known_clf = make_clf("tfidf", cfg.seed).fit(X_pool[keep], y_pool[keep])
    conf_known = known_clf.predict_proba(X_test).max(axis=1)
    flagged = conf_known < cfg.emerging_conf_threshold
    is_new = np.isin(y_test, holdout)
    emerging = {
        "holdout_intents": holdout,
        "confidence_threshold": cfg.emerging_conf_threshold,
        "flagged_share_of_new_intent_tickets": float(flagged[is_new].mean()),
        "flagged_share_of_known_intent_tickets": float(flagged[~is_new].mean()),
        "n_flagged": int(flagged.sum()),
        "new_intent_share_of_flagged": float(is_new[flagged].mean()) if flagged.any() else 0.0,
        "per_intent": {},
    }
    if flagged.sum() > cfg.min_cluster_size_emerging * 2:
        fi = np.flatnonzero(flagged)
        Uf = umap_reduce(Zt[fi], 5, cfg.umap_neighbors, cfg.seed)
        clf_lab = cluster(Uf, cfg.min_cluster_size_emerging)
        etab = cluster_table(test["core"].iloc[fi].tolist(), y_test[fi], clf_lab)
        pd.DataFrame([{**r, "keywords": ", ".join(r["keywords"])} for r in etab]).to_csv(results_dir / "emerging_clusters.csv", index=False)
        emerging["n_clusters_on_flagged"] = len(etab)
        for h in holdout:
            n_h = int((y_test == h).sum())
            best = None
            for r in etab:
                m = clf_lab == r["cluster"]
                hits = int((y_test[fi][m] == h).sum())
                if best is None or hits > best["hits"]:
                    best = {"cluster": r["cluster"], "hits": hits, "size": r["size"], "keywords": r["keywords"]}
            own = [r for r in etab if r["majority_intent"] == h]
            in_own = int(sum(((clf_lab == r["cluster"]) & (y_test[fi] == h)).sum() for r in own))
            own_size = int(sum(r["size"] for r in own))
            if best:
                emerging["per_intent"][h] = {
                    "clusters_with_this_majority": len(own),
                    "recall_in_those_clusters": in_own / max(n_h, 1),
                    "purity_of_those_clusters": in_own / max(own_size, 1),
                    "test_tickets": n_h,
                    "best_cluster_size": best["size"],
                    "best_cluster_recall": best["hits"] / max(n_h, 1),
                    "best_cluster_purity": best["hits"] / max(best["size"], 1),
                    "best_cluster_keywords": best["keywords"],
                }
    topics["emerging"] = emerging
    U2 = umap_reduce(Zt, 2, cfg.umap_neighbors, cfg.seed)
    hl = np.full(len(y_test), -1)
    for k, h in enumerate(holdout):
        hl[y_test == h] = k
    plots.umap_scatter(U2, cl, hl, holdout, fig_dir / "topics_umap.png")
    timings["topics_s"] = time.time() - t
    _log(f"topics: {topics['quality']}", t0)

    # 8. Figures, metrics, report -----------------------------------------------------------
    plots.confusion(
        baselines["tfidf_al_margin_2pct"]["confusion"],
        labels,
        "Confusion matrix: TF-IDF + LR, 2% labels chosen by margin AL (test)",
        fig_dir / "confusion_al_2pct.png",
    )
    plots.confusion(
        baselines["tfidf_full_labels"]["confusion"],
        labels,
        "Confusion matrix: TF-IDF + LR, 100% labels (test)",
        fig_dir / "confusion_full.png",
    )
    plots.per_intent_f1(
        {
            "Random 2%": {k: v["f1"] for k, v in baselines["tfidf_random_2pct"]["per_intent"].items()},
            "Margin AL 2%": {k: v["f1"] for k, v in baselines["tfidf_al_margin_2pct"]["per_intent"].items()},
            "100% labels": {k: v["f1"] for k, v in baselines["tfidf_full_labels"]["per_intent"].items()},
        },
        fig_dir / "per_intent_f1.png",
    )
    plots.pii_bars(
        {
            "spaCy NER + regex + cues": {k: v["recall_any_label"] for k, v in pii_main["per_type"].items()},
            "spaCy NER + regex": {k: v["recall_any_label"] for k, v in ablations["ner_plus_regex_no_cues"]["per_type"].items()},
            "regex + cues (no NER)": {k: v["recall_any_label"] for k, v in ablations["regex_and_cues_no_ner"]["per_type"].items()},
        },
        fig_dir / "pii_recall.png",
    )
    per_intent = pd.DataFrame(
        {
            m: {k: v["f1"] for k, v in baselines[m]["per_intent"].items()}
            for m in ("tfidf_full_labels", "emb_full_labels", "tfidf_random_2pct", "tfidf_al_margin_2pct", "emb_random_2pct")
        }
    )
    per_intent.index.name = "intent"
    per_intent["support"] = [baselines["tfidf_full_labels"]["per_intent"][k]["support"] for k in per_intent.index]
    per_intent.round(4).to_csv(results_dir / "per_intent_f1.csv")
    for m in ("tfidf_full_labels", "tfidf_al_margin_2pct"):
        pd.DataFrame(baselines[m]["confusion"], index=labels, columns=labels).to_csv(results_dir / f"confusion_{m}.csv")

    timings["total_s"] = time.time() - t0
    metrics = {
        "run": cfg.name,
        "seed": cfg.seed,
        "python": platform.python_version(),
        "data": {
            **prov,
            "n_tickets": int(len(df)),
            "n_intents": len(labels),
            "pii_ticket_share": float((df["pii_spans"] != "[]").mean()),
            "snippet_ticket_share": float(df["has_snippet"].mean()),
        },
        "pii": {
            "generator_labelled": pii_main,
            "ablations": ablations,
            "hand_labelled": pii_hand,
            "spacy_ner_available": red.has_ner,
            "synthetic_email_domains_left_after_redaction": leaked,
            "criterion_recall_ge_0_95_met": bool(pii_main["recall_any_label"] >= 0.95 and pii_hand["recall_any_label"] >= 0.95),
        },
        "embedder": {"backend": emb.backend, "note": emb.note},
        "classification": {k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in baselines.items()},
        "active_learning": al,
        "abstention": abst,
        "topics": topics,
        "timings_s": timings,
    }
    metrics = _r(metrics)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (results_dir / "report.md").write_text(render_report(metrics, table, labels, baselines))
    _log("done", t0)
    return metrics


def render_report(m: dict, table: list[dict], labels: list[str], baselines: dict) -> str:
    al = m["active_learning"]
    L = [
        "# Support ticket intelligence: run report",
        "",
        f"Data: {m['data']['data_source']} ({m['data']['n_tickets']} tickets, {m['data']['n_intents']} intents). All PII is SYNTHETIC.",
        "",
        "## Per-intent F1 (test)",
        "",
        "| Intent | Support | TF-IDF 100% | Embedding 100% | Random 2% | Margin AL 2% |",
        "|---|---|---|---|---|---|",
    ]
    for lab in labels:
        L.append(
            f"| {lab} | {baselines['tfidf_full_labels']['per_intent'][lab]['support']} "
            f"| {baselines['tfidf_full_labels']['per_intent'][lab]['f1']:.3f} "
            f"| {baselines['emb_full_labels']['per_intent'][lab]['f1']:.3f} "
            f"| {baselines['tfidf_random_2pct']['per_intent'][lab]['f1']:.3f} "
            f"| {baselines['tfidf_al_margin_2pct']['per_intent'][lab]['f1']:.3f} |"
        )
    L += [
        "",
        "## Active learning (TF-IDF, mean over repeats)",
        "",
        "| Strategy | F1 @ 2% | F1 @ 1% | Labels to match random @ 2% | Share of budget |",
        "|---|---|---|---|---|",
    ]
    for s, v in al["by_features"]["tfidf"]["strategies"].items():
        L.append(
            f"| {s} | {v['f1_at_budget_mean']:.4f} | {v['f1_at_half_budget_mean']:.4f} | {v['labels_to_match_random_at_budget']} | {v['label_share_to_match_random']} |"
        )
    L += ["", "## Topic clusters (test set)", "", "| Cluster | Size | Majority intent | Purity | Keywords |", "|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['cluster']} | {r['size']} | {r['majority_intent']} | {r['purity']:.2f} | {', '.join(r['keywords'])} |")
    return "\n".join(L) + "\n"
