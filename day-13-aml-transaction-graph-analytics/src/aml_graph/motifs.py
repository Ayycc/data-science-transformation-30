"""Label-free typology detectors built on time-respecting (temporal) motifs.

A *flow link* joins an incoming transfer e1 = (u -> v, t1, a1) to an onward transfer
e2 = (v -> w, t2, a2) when 0 < t2 - t1 <= 72 h and 0.8 <= a2 / a1 <= 1.0: money that
arrives and leaves again quickly, almost intact.  Flow links are temporal 2-path motifs;
chaining them gives temporal paths (mule chains) and temporal cycles, and grouping them by
(u, w) gives scatter-gather.  Fan-in / fan-out are temporal star motifs with a pass-through
condition.  None of the detectors look at labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DetectorConfig


@dataclass
class Detection:
    typology: str
    accounts: frozenset[int]
    tx_ids: frozenset[int]
    hub: int | None = None  # central account (collector / distributor / source)
    extra: dict = field(default_factory=dict)


def material(tx: pd.DataFrame, cfg: DetectorConfig) -> pd.DataFrame:
    return tx.loc[tx["amount"] >= cfg.min_amount, ["tx_id", "src", "dst", "ts", "amount"]].reset_index(drop=True)


def flow_links(mat: pd.DataFrame, cfg: DetectorConfig, n_chunks: int = 16) -> pd.DataFrame:
    """All temporal 2-paths (u -> v -> w) satisfying the hold-time and amount-ratio constraints."""
    e1 = mat.rename(columns={"tx_id": "tx_in", "src": "u", "dst": "v", "ts": "t1", "amount": "a1"})
    e2 = mat.rename(columns={"tx_id": "tx_out", "src": "v", "dst": "w", "ts": "t2", "amount": "a2"})
    parts = []
    for c in range(n_chunks):
        left = e1[e1["v"] % n_chunks == c]
        right = e2[e2["v"] % n_chunks == c]
        j = left.merge(right, on="v", how="inner")
        dt = j["t2"] - j["t1"]
        r = j["a2"] / j["a1"]
        keep = (dt > 0) & (dt <= cfg.flow_window_s) & (r >= cfg.flow_ratio_lo) & (r <= cfg.flow_ratio_hi) & (j["w"] != j["u"])
        parts.append(j.loc[keep])
    links = pd.concat(parts, ignore_index=True)
    links["hold_s"] = links["t2"] - links["t1"]
    return links[["tx_in", "tx_out", "u", "v", "w", "t1", "t2", "a1", "a2", "hold_s"]]


def _paths(links: pd.DataFrame, mat: pd.DataFrame, cfg: DetectorConfig) -> tuple[list[Detection], list[Detection]]:
    """Enumerate temporal cycles and maximal temporal chains over the flow-link DAG."""
    succ: dict[int, list[int]] = defaultdict(list)
    for a, b in zip(links["tx_in"].to_numpy(), links["tx_out"].to_numpy(), strict=True):
        succ[int(a)].append(int(b))
    has_pred = set(links["tx_out"].astype(int).tolist())
    involved = set(succ) | has_pred
    sub = mat[mat["tx_id"].isin(involved)]
    src = dict(zip(sub["tx_id"].astype(int), sub["src"].astype(int), strict=True))
    dst = dict(zip(sub["tx_id"].astype(int), sub["dst"].astype(int), strict=True))

    cycles: dict[frozenset, Detection] = {}
    chains: list[Detection] = []

    # temporal cycles: start at every edge with successors, return to the start account
    for e0 in succ:
        start = src[e0]
        stack = [(e0, (e0,), (start, dst[e0]))]
        found = 0
        while stack and found < cfg.max_paths_per_edge:
            e, path, accs = stack.pop()
            if len(path) >= cfg.cycle_max_hops:
                continue
            for nxt in succ.get(e, ()):
                d = dst[nxt]
                if d == start and len(path) + 1 >= 3:
                    key = frozenset(path + (nxt,))
                    if key not in cycles:
                        cycles[key] = Detection("cycle", frozenset(accs), key, hub=start)
                    found += 1
                elif d not in accs:
                    stack.append((nxt, path + (nxt,), accs + (d,)))

    # maximal chains: DFS from roots (edges without a predecessor link)
    roots = [e for e in succ if e not in has_pred]
    for e0 in roots:
        stack = [(e0, (e0,), (src[e0], dst[e0]))]
        emitted = 0
        while stack and emitted < cfg.max_paths_per_edge:
            e, path, accs = stack.pop()
            ext = [n for n in succ.get(e, ()) if dst[n] not in accs]
            if not ext:
                if len(path) >= cfg.chain_min_hops:
                    chains.append(Detection("mule_chain", frozenset(accs), frozenset(path), hub=accs[0]))
                    emitted += 1
                continue
            for nxt in ext:
                stack.append((nxt, path + (nxt,), accs + (dst[nxt],)))

    cyc = list(cycles.values())
    cyc_accs = [c.accounts for c in cyc]
    # a chain fully contained in a detected cycle is the same structure
    chains = [c for c in chains if not any(c.accounts <= ca for ca in cyc_accs)]
    return cyc, chains


def scatter_gather(links: pd.DataFrame, cfg: DetectorConfig) -> list[Detection]:
    g = links.groupby(["u", "w"])
    out = []
    for (u, w), d in g:
        mids = set(d["v"].astype(int).tolist())
        if len(mids) >= cfg.sg_min_intermediates:
            tx = frozenset(d["tx_in"].astype(int).tolist() + d["tx_out"].astype(int).tolist())
            out.append(
                Detection("scatter_gather", frozenset(mids | {int(u), int(w)}), tx, hub=int(u), extra={"sink": int(w), "n_mid": len(mids)})
            )
    return out


def window_counts(mat: pd.DataFrame, node_col: str, window: float) -> pd.Series:
    """Max number of material transfers on one side of an account inside any trailing window (vectorised)."""
    d = mat[[node_col, "ts"]].sort_values([node_col, "ts"])
    node = d[node_col].to_numpy()
    ts = d["ts"].to_numpy()
    key = node.astype(np.float64) * 1e9 + ts  # ts < 1e9 s, so keys are ordered by (node, ts)
    lo = np.searchsorted(key, key - window, side="left")
    cnt = np.arange(len(key)) - lo + 1
    return pd.Series(cnt, index=node).groupby(level=0).max()


def _stars(mat: pd.DataFrame, cfg: DetectorConfig, kind: str) -> list[Detection]:
    """fan_in: many senders -> hub, then hub forwards; fan_out: hub receives, then pays many."""
    hub_col, cp_col = ("dst", "src") if kind == "fan_in" else ("src", "dst")
    other_col = "src" if kind == "fan_in" else "dst"  # flows on the other side of the hub
    wc = window_counts(mat, hub_col, cfg.star_window_s)
    cand = set(wc.index[wc >= cfg.star_min_counterparties].tolist())
    side = mat[mat[hub_col].isin(cand)].sort_values([hub_col, "ts"])
    opp = mat[mat[other_col].isin(cand)].sort_values([other_col, "ts"])
    opp_groups = {k: (v["ts"].to_numpy(), np.cumsum(v["amount"].to_numpy())) for k, v in opp.groupby(other_col)}
    out = []
    for hub, d in side.groupby(hub_col):
        ts, cps, amt, ids = d["ts"].to_numpy(), d[cp_col].to_numpy(), d["amount"].to_numpy(), d["tx_id"].to_numpy()
        if hub not in opp_groups:
            continue
        ots, ocum = opp_groups[hub]
        counter: Counter = Counter()
        i, total = 0, 0.0
        hit_cps: set[int] = set()
        hit_tx: set[int] = set()
        for j in range(len(ts)):
            counter[cps[j]] += 1
            total += amt[j]
            while ts[i] < ts[j] - cfg.star_window_s:
                counter[cps[i]] -= 1
                if counter[cps[i]] == 0:
                    del counter[cps[i]]
                total -= amt[i]
                i += 1
            if len(counter) < cfg.star_min_counterparties or total < cfg.star_min_total:
                continue
            if kind == "fan_in":
                lo_t, hi_t = ts[i], ts[j] + cfg.star_follow_s
            else:
                lo_t, hi_t = ts[i] - cfg.star_follow_s, ts[j]
            a = np.searchsorted(ots, lo_t, side="left")
            b = np.searchsorted(ots, hi_t, side="right")
            flow = (ocum[b - 1] if b > 0 else 0.0) - (ocum[a - 1] if a > 0 else 0.0)
            if flow >= cfg.star_pass_through * total:
                hit_cps.update(int(c) for c in cps[i : j + 1])
                hit_tx.update(int(t) for t in ids[i : j + 1])
        if hit_cps:
            out.append(Detection(kind, frozenset(hit_cps | {int(hub)}), frozenset(hit_tx), hub=int(hub), extra={"n_cp": len(hit_cps)}))
    return out


def detect_all(tx: pd.DataFrame, cfg: DetectorConfig = DetectorConfig()) -> tuple[dict[str, list[Detection]], pd.DataFrame, pd.DataFrame]:
    """Run every typology detector. Returns detections, flow links and the material-edge table."""
    mat = material(tx, cfg)
    links = flow_links(mat, cfg)
    cycles, chains = _paths(links, mat, cfg)
    det = {
        "fan_in": _stars(mat, cfg, "fan_in"),
        "fan_out": _stars(mat, cfg, "fan_out"),
        "cycle": cycles,
        "scatter_gather": scatter_gather(links, cfg),
        "mule_chain": chains,
    }
    return det, links, mat


def motif_node_features(
    n_nodes: int, det: dict[str, list[Detection]], links: pd.DataFrame, mat: pd.DataFrame, cfg: DetectorConfig
) -> pd.DataFrame:
    """Per-account temporal motif counts used as model features."""
    f = pd.DataFrame(index=pd.RangeIndex(n_nodes, name="account_id"))
    for typ, ds in det.items():
        cnt = np.zeros(n_nodes)
        hub = np.zeros(n_nodes)
        for d in ds:
            cnt[list(d.accounts)] += 1
            if d.hub is not None:
                hub[d.hub] += 1
        f[f"motif_{typ}"] = cnt
        if typ in ("fan_in", "fan_out", "scatter_gather"):
            f[f"motif_{typ}_hub"] = hub
    piv = links.groupby("v")
    f["flow_links_through"] = piv.size().reindex(f.index, fill_value=0).to_numpy()
    f["flow_median_hold_h"] = (piv["hold_s"].median() / 3600).reindex(f.index).fillna(-1).to_numpy()
    f["flow_out_links"] = links.groupby("u").size().reindex(f.index, fill_value=0).to_numpy()
    f["max_in_72h"] = window_counts(mat, "dst", cfg.star_window_s).reindex(f.index, fill_value=0).to_numpy()
    f["max_out_72h"] = window_counts(mat, "src", cfg.star_window_s).reindex(f.index, fill_value=0).to_numpy()
    f["motif_any"] = (f[[f"motif_{t}" for t in det]].sum(axis=1) > 0).astype(float)
    return f


def evaluate_detectors(det: dict[str, list[Detection]], instances: pd.DataFrame, labels: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Instance-level recall per typology and account-level precision of each detector.

    An instance counts as detected by its typology's detector when:
    * fan_in / fan_out: a detection is hubbed on the instance's collector / distributor;
    * cycle: a detected cycle's accounts all belong to the instance ring and it closes >= 3 of them;
    * scatter_gather: a detection is sourced at the instance source, sinks at its sink;
    * mule_chain: a detected chain shares >= 4 accounts (or >= 50%) with the instance chain.
    """
    rows = []
    for (iid, typ), m in instances.groupby(["instance_id", "typology"]):
        accs = set(m["account_id"].astype(int).tolist())
        role = dict(zip(m["scheme_role"], m["account_id"].astype(int), strict=False))
        slow = bool(m["slow"].iloc[0])
        ds = det[typ]
        if typ == "fan_in":
            hit = any(d.hub == role["collector"] for d in ds)
        elif typ == "fan_out":
            hit = any(d.hub == role["distributor"] for d in ds)
        elif typ == "cycle":
            hit = any(d.accounts <= accs and len(d.accounts) >= 3 for d in ds)
        elif typ == "scatter_gather":
            hit = any(d.hub == role["source"] and d.extra.get("sink") == role["sink"] for d in ds)
        else:
            hit = any(len(d.accounts & accs) >= min(4, max(2, len(accs) // 2)) for d in ds)
        any_hit = any(len(d.accounts & accs) >= 2 for dl in det.values() for d in dl)
        rows.append(
            {"instance_id": iid, "typology": typ, "slow": slow, "n_accounts": len(accs), "detected": hit, "any_detector_touch": any_hit}
        )
    per_inst = pd.DataFrame(rows)
    report = {}
    for typ in det:
        sub = per_inst[per_inst["typology"] == typ]
        flagged = set().union(*[d.accounts for d in det[typ]]) if det[typ] else set()
        fl = np.array(sorted(flagged), dtype=int)
        report[typ] = {
            "instances": int(len(sub)),
            "recall": float(sub["detected"].mean()) if len(sub) else 0.0,
            "recall_fast": float(sub.loc[~sub["slow"], "detected"].mean()) if (~sub["slow"]).any() else 0.0,
            "recall_slow": float(sub.loc[sub["slow"], "detected"].mean()) if sub["slow"].any() else 0.0,
            "n_detections": int(len(det[typ])),
            "flagged_accounts": int(len(fl)),
            "account_precision": float(labels[fl].mean()) if len(fl) else 0.0,
            "hub_precision": float(np.mean([labels[d.hub] for d in det[typ] if d.hub is not None])) if det[typ] else 0.0,
        }
    all_flag = set().union(*[d.accounts for dl in det.values() for d in dl])
    fl = np.array(sorted(all_flag), dtype=int)
    report["all_detectors"] = {
        "flagged_accounts": int(len(fl)),
        "account_precision": float(labels[fl].mean()) if len(fl) else 0.0,
        "account_recall": float(labels[fl].sum() / labels.sum()) if labels.sum() else 0.0,
        "instance_recall_own_detector": float(per_inst["detected"].mean()),
        "instance_touch_any_detector": float(per_inst["any_detector_touch"].mean()),
    }
    return report, per_inst
