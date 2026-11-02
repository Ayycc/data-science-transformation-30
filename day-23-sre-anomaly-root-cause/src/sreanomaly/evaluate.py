"""Evaluation against the injected-incident ledger: matching, detection precision/recall/delay."""

from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_SLACK = 15  # minutes either side of the injected fault window


def match_groups(groups: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    """Match each alert group to the injected incidents it covers: time overlap with
    [start - slack, end + slack] AND at least one alerting service in the incident's affected set.
    `incident_ids` lists every covered incident (a group can cover two concurrent incidents);
    `incident_id` is the best one (largest overlap) or -1 for a false-positive group."""
    out = groups.copy()
    ids, best_ids = [], []
    for g in out.itertuples(index=False):
        svcs = set(g.services.split(";"))
        hits = []
        for inc in incidents.itertuples(index=False):
            a, b = inc.start - MATCH_SLACK, inc.end + MATCH_SLACK
            ov = min(b, g.end) - max(a, g.start) + 1
            if ov > 0 and svcs & set(inc.affected.split(";")):
                hits.append((ov, inc.incident_id))
        ids.append([i for _, i in sorted(hits, reverse=True)])
        best_ids.append(max(hits)[1] if hits else -1)
    out["incident_ids"] = ids
    out["incident_id"] = best_ids
    return out


def detection_metrics(matched: pd.DataFrame, incidents: pd.DataFrame) -> dict:
    n_groups = len(matched)
    tp_groups = int((matched["incident_id"] >= 0).sum())
    cover = matched[["start", "incident_ids"]].explode("incident_ids").dropna()
    detected = cover.groupby("incident_ids")["start"].min()
    delays = np.array([detected[i] - incidents.set_index("incident_id").at[i, "start"] for i in detected.index], dtype=float)
    precision = tp_groups / n_groups if n_groups else 0.0
    recall = len(detected) / len(incidents) if len(incidents) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    frag = cover.groupby("incident_ids").size()
    return dict(
        n_groups=n_groups,
        true_positive_groups=tp_groups,
        false_positive_groups=n_groups - tp_groups,
        precision=round(precision, 4),
        incidents_detected=int(len(detected)),
        recall=round(recall, 4),
        f1=round(f1, 4),
        delay_median_min=float(np.median(delays)) if len(delays) else None,
        delay_mean_min=round(float(np.mean(delays)), 2) if len(delays) else None,
        delay_p90_min=round(float(np.percentile(delays, 90)), 1) if len(delays) else None,
        groups_per_detected_incident=round(float(frag.mean()), 3) if len(frag) else None,
    )
