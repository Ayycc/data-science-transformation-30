"""20-service microservice call graph (SYNTHETIC). Edges point caller -> callee."""

from __future__ import annotations

import networkx as nx

SERVICES: list[str] = [
    "web",
    "mobile-bff",
    "api-gateway",
    "auth",
    "catalog",
    "cart",
    "checkout",
    "search",
    "user-profile",
    "payment",
    "inventory",
    "shipping",
    "order",
    "fraud",
    "notification",
    "cache",
    "catalog-db",
    "order-db",
    "user-db",
    "message-queue",
]

CALLS: list[tuple[str, str]] = [
    ("web", "api-gateway"),
    ("mobile-bff", "api-gateway"),
    ("api-gateway", "auth"),
    ("api-gateway", "catalog"),
    ("api-gateway", "cart"),
    ("api-gateway", "checkout"),
    ("api-gateway", "search"),
    ("api-gateway", "user-profile"),
    ("checkout", "payment"),
    ("checkout", "inventory"),
    ("checkout", "shipping"),
    ("checkout", "order"),
    ("payment", "fraud"),
    ("fraud", "user-db"),
    ("auth", "user-db"),
    ("user-profile", "user-db"),
    ("catalog", "catalog-db"),
    ("catalog", "cache"),
    ("search", "cache"),
    ("cart", "cache"),
    ("inventory", "catalog-db"),
    ("order", "order-db"),
    ("order", "message-queue"),
    ("shipping", "notification"),
    ("notification", "message-queue"),
]


def call_graph() -> nx.DiGraph:
    """Directed call graph, caller -> callee. Faults propagate against the edges (callee -> caller)."""
    g = nx.DiGraph()
    g.add_nodes_from(SERVICES)
    g.add_edges_from(CALLS)
    assert nx.is_directed_acyclic_graph(g)
    return g


def callees_first_order(g: nx.DiGraph) -> list[str]:
    """Topological order with callees before callers (so caller latency can include callee latency)."""
    return list(reversed(list(nx.topological_sort(g))))


def upstream_depths(g: nx.DiGraph, root: str) -> dict[str, int]:
    """Services affected by a fault at `root` (root and all transitive callers) with hop distance."""
    return nx.single_source_shortest_path_length(g.reverse(copy=False), root)


def hop_distance_undirected(g: nx.DiGraph) -> dict[str, dict[str, int]]:
    return dict(nx.all_pairs_shortest_path_length(g.to_undirected(as_view=True)))
