"""Explicit causal DAG, identification with DoWhy and adjustment-set checks with NetworkX."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

TREATMENT = "treated"
OUTCOMES = ["earnings_post", "employed_post"]
UNOBSERVED = {"ability"}

EDGES: list[tuple[str, str]] = [
    # demographics -> labour-market history
    ("age", "married"),
    ("age", "unemployed_base"),
    ("educ", "unemployed_base"),
    ("urban", "unemployed_base"),
    ("age", "prior_earn_2"),
    ("educ", "prior_earn_2"),
    ("urban", "prior_earn_2"),
    ("female", "prior_earn_2"),
    ("prior_earn_2", "prior_earn_1"),
    ("unemployed_base", "prior_earn_1"),
    ("age", "prior_earn_1"),
    ("educ", "prior_earn_1"),
    ("urban", "prior_earn_1"),
    ("female", "prior_earn_1"),
    # enrolment (selection)
    ("age", TREATMENT),
    ("female", TREATMENT),
    ("educ", TREATMENT),
    ("urban", TREATMENT),
    ("unemployed_base", TREATMENT),
    ("prior_earn_1", TREATMENT),
    ("distance_km", TREATMENT),
    # mediator / collider
    (TREATMENT, "certificate"),
    ("educ", "certificate"),
    ("ability", "certificate"),
    # outcomes
    (TREATMENT, "earnings_post"),
    (TREATMENT, "employed_post"),
    ("certificate", "earnings_post"),
    ("certificate", "employed_post"),
    ("ability", "earnings_post"),
    ("ability", "employed_post"),
]
for _v in ["age", "female", "educ", "married", "urban", "unemployed_base", "prior_earn_1", "prior_earn_2"]:
    EDGES.append((_v, "earnings_post"))
for _v in ["age", "educ", "urban", "unemployed_base", "prior_earn_1"]:
    EDGES.append((_v, "employed_post"))


def build_dag() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(EDGES)
    for n in g.nodes:
        g.nodes[n]["observed"] = "no" if n in UNOBSERVED else "yes"
    assert nx.is_directed_acyclic_graph(g)
    return g


def _backdoor_graph(g: nx.DiGraph, treatment: str) -> nx.DiGraph:
    h = g.copy()
    h.remove_edges_from(list(g.out_edges(treatment)))
    return h


def is_valid_adjustment(g: nx.DiGraph, treatment: str, outcome: str, z: set[str]) -> bool:
    """Pearl's backdoor criterion: no descendant of T in Z and Z d-separates T from Y once T's out-edges are cut."""
    if z & nx.descendants(g, treatment):
        return False
    if z & UNOBSERVED:
        return False
    return nx.is_d_separator(_backdoor_graph(g, treatment), {treatment}, {outcome}, z)


def minimal_adjustment(g: nx.DiGraph, treatment: str, outcome: str) -> set[str]:
    h = _backdoor_graph(g, treatment)
    observed = {n for n in g.nodes if n not in UNOBSERVED} - nx.descendants(g, treatment) - {treatment, outcome}
    z = nx.find_minimal_d_separator(h, {treatment}, {outcome}, included=set(), restricted=observed)
    return set(z) if z is not None else set()


def dowhy_identification(df: pd.DataFrame, outcome: str = "earnings_post") -> dict[str, object]:
    """Identify the effect with DoWhy on the explicit graph; returns the estimand summary."""
    from dowhy import CausalModel

    g = build_dag()
    cols = [c for c in df.columns if c in g.nodes]
    model = CausalModel(data=df[cols], treatment=TREATMENT, outcome=outcome, graph=g)
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    return {
        "model": model,
        "estimand": estimand,
        "backdoor_variables": sorted(estimand.get_backdoor_variables()),
        "instrumental_variables": sorted(estimand.get_instrumental_variables()),
        "frontdoor_variables": sorted(estimand.get_frontdoor_variables() or []),
    }


def adjustment_report(confounders: list[str]) -> dict[str, object]:
    g = build_dag()
    out: dict[str, object] = {}
    for y in OUTCOMES:
        out[y] = {
            "minimal_set": sorted(minimal_adjustment(g, TREATMENT, y)),
            "chosen_set_valid": is_valid_adjustment(g, TREATMENT, y, set(confounders)),
            "chosen_plus_certificate_valid": is_valid_adjustment(g, TREATMENT, y, set(confounders) | {"certificate"}),
            "empty_set_valid": is_valid_adjustment(g, TREATMENT, y, set()),
        }
    out["descendants_of_treatment"] = sorted(nx.descendants(g, TREATMENT))
    out["n_nodes"] = g.number_of_nodes()
    out["n_edges"] = g.number_of_edges()
    return out


_POS = {
    "age": (0, 6),
    "female": (0, 5),
    "educ": (0, 4),
    "urban": (0, 3),
    "married": (0, 1.5),
    "distance_km": (0, 0),
    "unemployed_base": (1.6, 5.5),
    "prior_earn_2": (1.6, 3.5),
    "prior_earn_1": (3.0, 4.5),
    TREATMENT: (4.4, 1.5),
    "certificate": (6.0, 0.3),
    "ability": (7.6, -0.4),
    "earnings_post": (7.6, 4.2),
    "employed_post": (7.6, 2.4),
}


def draw_dag(path: Path) -> None:
    g = build_dag()
    colors = []
    for n in g.nodes:
        if n == TREATMENT:
            colors.append("#d9822b")
        elif n in OUTCOMES:
            colors.append("#2b7bd9")
        elif n in UNOBSERVED:
            colors.append("#bbbbbb")
        elif n in ("certificate",):
            colors.append("#c0392b")
        elif n == "distance_km":
            colors.append("#8e44ad")
        else:
            colors.append("#7fb77e")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    edge_colors = ["#d9822b" if u == TREATMENT else "#999999" for u, _ in g.edges]
    nx.draw_networkx_edges(g, _POS, ax=ax, edge_color=edge_colors, arrows=True, arrowsize=12, width=1.0, node_size=2600, alpha=0.8)
    nx.draw_networkx_nodes(g, _POS, ax=ax, node_color=colors, node_size=2600, alpha=0.95)
    nx.draw_networkx_labels(g, _POS, ax=ax, font_size=8)
    ax.set_title(
        "Causal DAG for programme enrolment (SYNTHETIC study)\n"
        "green = adjust (confounders), purple = instrument-like (enrolment only), red = mediator/collider (do NOT adjust), grey = unobserved",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
