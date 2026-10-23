"""Seeded SYNTHETIC transaction-graph generator with injected laundering typologies.

Background activity mimics a retail bank: person-to-person transfers, card purchases at
merchants, business-to-business invoices, payroll runs and merchant settlements, mostly
inside a planted geography (regions).  Five laundering typologies are injected on top of
existing accounts (so laundering accounts also carry normal camouflage traffic) and on
freshly opened mule accounts:

* ``fan_in``         many mules -> one collector -> exit account(s)
* ``fan_out``        origin -> one distributor -> many recipients
* ``cycle``          A -> B -> ... -> A, money returns to the start minus fees
* ``scatter_gather`` source -> k intermediates -> one sink
* ``mule_chain``     a linear chain of quick pass-through hops

A share of each typology is "slow" (hold times beyond the 72 h detector window) or uses
amounts outside the structuring band, so rule recall is honestly below 100%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DAY, HOUR, TYPOLOGIES, SimConfig

ROLES = np.array(["personal", "business", "merchant"])


@dataclass
class SimulatedBank:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    instances: pd.DataFrame  # one row per (instance, account) with role in the scheme


class _LocalSampler:
    """Draw an account of a given role, either inside a region (uniform) or globally (weighted)."""

    def __init__(self, ids: np.ndarray, regions: np.ndarray, weights: np.ndarray, n_regions: int):
        order = np.argsort(regions, kind="stable")
        self.ids = ids[order]
        self.counts = np.bincount(regions, minlength=n_regions)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self.global_ids = ids
        p = weights / weights.sum()
        self.global_p = p

    def draw(self, rng: np.random.Generator, regions: np.ndarray, p_local: float) -> np.ndarray:
        n = len(regions)
        out = rng.choice(self.global_ids, size=n, p=self.global_p)
        local = rng.random(n) < p_local
        cnt = self.counts[regions]
        ok = local & (cnt > 0)
        pick = self.offsets[regions[ok]] + (rng.random(ok.sum()) * cnt[ok]).astype(np.int64)
        out[ok] = self.ids[pick]
        return out


def _accounts(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_accounts
    role = rng.choice(3, size=n, p=[0.875, 0.09, 0.035])
    # Zipf-like region sizes
    region_p = 1.0 / np.arange(1, cfg.n_regions + 1) ** 0.6
    region_p /= region_p.sum()
    region = rng.choice(cfg.n_regions, size=n, p=region_p)
    region_risk = rng.choice(3, size=cfg.n_regions, p=[0.7, 0.22, 0.08])
    activity = rng.lognormal(0.0, 1.0, size=n)
    activity[role == 2] *= 6.0  # merchants are popular
    payroll = (role == 1) & (rng.random(n) < 0.3)
    open_day = np.where(rng.random(n) < 0.10, rng.uniform(0, cfg.n_days * 0.8, size=n), 0.0)
    return pd.DataFrame(
        {
            "account_id": np.arange(n, dtype=np.int64),
            "role": ROLES[role],
            "region": region.astype(np.int64),
            "region_risk": region_risk[region].astype(np.int64),
            "activity": activity,
            "is_payroll": payroll,
            "open_day": open_day,
        }
    )


def _background(cfg: SimConfig, acc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_tx = cfg.n_background_tx
    ids = acc["account_id"].to_numpy()
    reg = acc["region"].to_numpy()
    w = acc["activity"].to_numpy()
    role = acc["role"].to_numpy()
    groups = {
        "personal": role == "personal",
        "business": role == "business",
        "merchant": role == "merchant",
        "payroll": acc["is_payroll"].to_numpy(),
    }
    samplers = {k: _LocalSampler(ids[m], reg[m], w[m], cfg.n_regions) for k, m in groups.items() if m.any()}

    channels = {
        # name: (share, src group, dst group, p_local, log-mean, log-sd)
        "p2p": (0.26, "personal", "personal", 0.85, 4.5, 1.0),
        "large_transfer": (0.06, "personal", "personal", 0.60, 8.0, 1.1),
        "purchase": (0.43, "personal", "merchant", 0.70, 3.5, 0.9),
        "b2b": (0.12, "business", "business", 0.50, 8.0, 1.0),
        "payroll": (0.08, "payroll", "personal", 0.90, 7.8, 0.3),
        "settlement": (0.05, "merchant", "business", 0.60, 8.5, 0.8),
    }
    parts = []
    for name, (share, sg, dg, p_local, mu, sd) in channels.items():
        k = int(round(n_tx * share))
        src_s = samplers[sg]
        src = rng.choice(src_s.global_ids, size=k, p=src_s.global_p)
        dst = samplers[dg].draw(rng, reg[src], p_local)
        amount = np.round(rng.lognormal(mu, sd, size=k), 2)
        if name == "payroll":
            day = rng.integers(0, cfg.n_days // 15, size=k) * 15 + rng.integers(0, 2, size=k)
            ts = day * DAY + 9 * HOUR + rng.uniform(0, 4 * HOUR, size=k)
        else:
            day = rng.integers(0, cfg.n_days, size=k)
            # daytime-heavy intraday profile
            hour = np.clip(rng.normal(14, 4, size=k), 0, 23.99)
            ts = day * DAY + hour * HOUR + rng.uniform(0, 59 * 60, size=k)
        parts.append(pd.DataFrame({"src": src, "dst": dst, "ts": ts, "amount": amount, "channel": name}))
    tx = pd.concat(parts, ignore_index=True)
    tx = tx[tx["src"] != tx["dst"]].reset_index(drop=True)
    # respect account opening dates: move transactions before an account opened into its open period
    open_s = acc["open_day"].to_numpy() * DAY
    start = np.maximum(open_s[tx["src"].to_numpy()], open_s[tx["dst"].to_numpy()])
    bad = tx["ts"].to_numpy() < start
    horizon = cfg.n_days * DAY
    new_ts = start[bad] + rng.random(bad.sum()) * (horizon - start[bad])
    ts = tx["ts"].to_numpy().copy()
    ts[bad] = new_ts
    tx["ts"] = ts
    return tx


def _inject(cfg: SimConfig, acc: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = acc.loc[acc["role"] != "merchant"]
    # high-risk regions supply slightly more mules (a weak tabular signal)
    p = 1.0 + 0.8 * eligible["region_risk"].to_numpy()
    p = p / p.sum()
    n_inst = cfg.instances_per_typology
    need = n_inst * (17 + 17 + 8 + 12 + 9) + 50
    pool = rng.choice(eligible["account_id"].to_numpy(), size=min(need, len(eligible)), replace=False, p=p)
    it = iter(pool.tolist())

    def take(k: int) -> list[int]:
        return [next(it) for _ in range(k)]

    horizon = cfg.n_days * DAY
    rows: list[tuple] = []  # src, dst, ts, amount, typology, instance
    members: list[tuple] = []  # instance, typology, account, scheme role, slow
    inst_id = 0
    for typ in TYPOLOGIES:
        for _ in range(n_inst):
            slow = rng.random() < 0.20
            lag_hi = 120 * HOUR if slow else 48 * HOUR
            t0 = rng.uniform(10 * DAY, horizon - 12 * DAY)
            if typ == "fan_in":
                k = int(rng.integers(6, 16))
                senders, collector = take(k), take(1)[0]
                exits = take(int(rng.integers(1, 3)))
                structuring = rng.random() < 0.6
                amts = rng.uniform(3000, 9900, k) if structuring else rng.uniform(1000, 25000, k)
                ts = t0 + rng.uniform(0, lag_hi, k)
                rows += [(s, collector, t, a, typ, inst_id) for s, t, a in zip(senders, ts, amts, strict=True)]
                total = amts.sum() * rng.uniform(0.85, 0.95)
                t_exit = ts.max() + rng.uniform(6 * HOUR, 96 * HOUR)
                for e in exits:
                    rows.append((collector, e, t_exit + rng.uniform(0, 6 * HOUR), total / len(exits), typ, inst_id))
                members += [(inst_id, typ, s, "sender", slow) for s in senders]
                members += [(inst_id, typ, collector, "collector", slow)]
                members += [(inst_id, typ, e, "exit", slow) for e in exits]
            elif typ == "fan_out":
                k = int(rng.integers(6, 16))
                origin, dist = take(1)[0], take(1)[0]
                recips = take(k)
                big = rng.uniform(40_000, 150_000)
                rows.append((origin, dist, t0, big, typ, inst_id))
                share = rng.uniform(0.8, 1.2, k)
                amts = big * rng.uniform(0.85, 0.95) * share / share.sum()
                ts = t0 + rng.uniform(1 * HOUR, lag_hi, k)
                rows += [(dist, r, t, a, typ, inst_id) for r, t, a in zip(recips, ts, amts, strict=True)]
                members += [(inst_id, typ, origin, "origin", slow), (inst_id, typ, dist, "distributor", slow)]
                members += [(inst_id, typ, r, "recipient", slow) for r in recips]
            elif typ == "cycle":
                length = int(rng.integers(3, 8))
                ring = take(length)
                a, t = rng.uniform(20_000, 150_000), t0
                for i in range(length):
                    rows.append((ring[i], ring[(i + 1) % length], t, a, typ, inst_id))
                    t += rng.uniform(2 * HOUR, (120 if slow else 60) * HOUR)
                    a *= rng.uniform(0.93, 0.99)
                members += [(inst_id, typ, r, "ring", slow) for r in ring]
            elif typ == "scatter_gather":
                k = int(rng.integers(4, 11))
                src, sink = take(1)[0], take(1)[0]
                mids = take(k)
                big = rng.uniform(30_000, 120_000)
                for m in mids:
                    a1 = big / k * rng.uniform(0.9, 1.1)
                    t1 = t0 + rng.uniform(0, 24 * HOUR)
                    rows.append((src, m, t1, a1, typ, inst_id))
                    rows.append(
                        (m, sink, t1 + rng.uniform(4 * HOUR, (120 if slow else 60) * HOUR), a1 * rng.uniform(0.9, 0.98), typ, inst_id)
                    )
                members += [(inst_id, typ, src, "source", slow), (inst_id, typ, sink, "sink", slow)]
                members += [(inst_id, typ, m, "intermediate", slow) for m in mids]
            else:  # mule_chain
                hops = int(rng.integers(4, 9))
                chain = take(hops + 1)
                a, t = rng.uniform(5_000, 60_000), t0
                for i in range(hops):
                    rows.append((chain[i], chain[i + 1], t, a, typ, inst_id))
                    t += rng.uniform(1 * HOUR, lag_hi)
                    a *= rng.uniform(0.9, 0.99)
                members += [(inst_id, typ, c, "hop", slow) for c in chain]
            inst_id += 1
    tx = pd.DataFrame(rows, columns=["src", "dst", "ts", "amount", "typology", "instance_id"])
    tx["amount"] = tx["amount"].round(2)
    tx["channel"] = "transfer"
    inst = pd.DataFrame(members, columns=["instance_id", "typology", "account_id", "scheme_role", "slow"])
    return tx, inst


def simulate(cfg: SimConfig = SimConfig()) -> SimulatedBank:
    """Generate the SYNTHETIC bank. Fully determined by ``cfg.seed``."""
    rng = np.random.default_rng(cfg.seed)
    acc = _accounts(cfg, rng)
    bg = _background(cfg, acc, rng)
    lt, inst = _inject(cfg, acc, rng)

    # some mules are freshly opened accounts with little organic activity
    first_use = inst.merge(lt.groupby("instance_id")["ts"].min().rename("t_start"), on="instance_id")
    fresh = first_use.drop_duplicates("account_id")
    fresh = fresh[rng.random(len(fresh)) < 0.35]
    open_day = acc["open_day"].to_numpy().copy()
    open_day[fresh["account_id"].to_numpy()] = np.maximum(0.0, fresh["t_start"].to_numpy() / DAY - rng.uniform(1, 30, len(fresh)))
    acc["open_day"] = open_day
    # thin out organic traffic of fresh mules that falls before their opening date
    open_s = open_day * DAY
    keep = (bg["ts"].to_numpy() >= open_s[bg["src"].to_numpy()]) & (bg["ts"].to_numpy() >= open_s[bg["dst"].to_numpy()])
    bg = bg[keep]

    bg = bg.assign(typology="none", instance_id=-1)
    tx = pd.concat([bg, lt], ignore_index=True)
    tx["is_laundering"] = tx["typology"] != "none"
    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    tx.insert(0, "tx_id", np.arange(len(tx), dtype=np.int64))
    tx["src"] = tx["src"].astype(np.int64)
    tx["dst"] = tx["dst"].astype(np.int64)
    tx["instance_id"] = tx["instance_id"].astype(np.int64)

    first = inst.drop_duplicates("account_id")
    acc = acc.merge(first[["account_id", "typology", "instance_id"]], on="account_id", how="left")
    acc["typology"] = acc["typology"].fillna("none")
    acc["instance_id"] = acc["instance_id"].fillna(-1).astype(np.int64)
    acc["is_laundering"] = acc["typology"] != "none"
    acc = acc.drop(columns=["activity"])
    return SimulatedBank(accounts=acc, transactions=tx, instances=inst)
