"""Alert de-duplication: per-service alert episodes -> incidents by temporal + topological grouping."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from .topology import hop_distance_undirected


def group_alerts(ep: pd.DataFrame, g: nx.DiGraph, window: int = 10, max_hops: float = 2) -> pd.DataFrame:
    """Greedy online grouping in start-time order.

    An episode joins an open group if it starts no later than `window` minutes after the group's latest
    alert end AND its service is within `max_hops` (undirected call-graph hops) of a service already in the
    group. If it qualifies for several groups they are merged. max_hops=inf gives temporal-only grouping.
    Returns ep with a `group` column."""
    hops = hop_distance_undirected(g)
    ep = ep.sort_values(["start", "service"]).reset_index(drop=True).copy()
    groups: dict[int, dict] = {}
    closed: list[dict] = []
    assign = []
    nxt = 0
    for row in ep.itertuples(index=False):
        hits = []
        for gid, grp in groups.items():
            if row.start > grp["last_end"] + window:
                continue
            d = min(hops[row.service].get(s, math.inf) for s in grp["services"])
            if d <= max_hops:
                hits.append(gid)
        if not hits:
            gid = nxt
            nxt += 1
            groups[gid] = dict(services={row.service}, last_end=row.end, members=[])
        else:
            gid = hits[0]
            for other in hits[1:]:
                groups[gid]["services"] |= groups[other]["services"]
                groups[gid]["last_end"] = max(groups[gid]["last_end"], groups[other]["last_end"])
                groups[gid]["members"] += groups[other]["members"]
                del groups[other]
            groups[gid]["services"].add(row.service)
            groups[gid]["last_end"] = max(groups[gid]["last_end"], row.end)
        groups[gid]["members"].append(len(assign))
        assign.append(gid)
        # close groups that can no longer receive alerts (keeps the scan O(open groups))
        for k in [k for k, v in groups.items() if row.start > v["last_end"] + window and k != gid]:
            closed.append(groups.pop(k))
    label = [0] * len(assign)
    for new_id, grp in enumerate(sorted(closed + list(groups.values()), key=lambda v: min(v["members"]))):
        for m in grp["members"]:
            label[m] = new_id
    ep["group"] = label
    return ep


def summarise_groups(ep: pd.DataFrame) -> pd.DataFrame:
    agg = ep.groupby("group").agg(
        start=("start", "min"),
        end=("end", "max"),
        n_alerts=("service", "size"),
        services=("service", lambda s: ";".join(sorted(set(s)))),
        peak=("peak", "max"),
    )
    return agg.reset_index()
