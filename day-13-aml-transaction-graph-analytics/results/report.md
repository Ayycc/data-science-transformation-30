# Day 13 results report (SYNTHETIC data)

Data: IBM AML probe: URLError: <urlopen error Tunnel connection failed: 403 Forbidden>; fell back to seeded SYNTHETIC generator (seed 20261023)

## Graph

- n_accounts: 100000
- n_transactions: 1980990
- n_unique_pairs: 1623588
- span_days: 180.0
- total_amount: 3069396377.97
- density: 0.00016236042360423604
- reciprocity: 0.020752801819180728
- mean_in_degree: 16.23588
- max_in_degree: 2129
- max_out_degree: 774
- gini_in_degree: 0.5438007106852231
- largest_wcc_share: 0.99998
- largest_scc_share: 0.97952
- n_scc_nontrivial: 1
- isolated_accounts: 2

## Typology detectors (instance recall, label-free rules)

| Typology | Instances | Recall | Recall fast | Recall slow | Flagged accounts | Account precision |
|---|---|---|---|---|---|---|
| fan_in | 80 | 0.988 | 1.000 | 0.944 | 8,142 | 0.134 |
| fan_out | 80 | 1.000 | 1.000 | 1.000 | 37,810 | 0.052 |
| cycle | 80 | 0.750 | 1.000 | 0.231 | 323 | 0.991 |
| scatter_gather | 80 | 0.950 | 1.000 | 0.733 | 710 | 0.965 |
| mule_chain | 80 | 0.863 | 1.000 | 0.353 | 970 | 0.546 |

## Models (held-out accounts, grouped by scheme)

| Feature set | Features | PR-AUC | ROC-AUC | Precision@top1% | Alerts @80% recall | False alerts @80% recall |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.536 | 0.886 | 0.970 | 7,704 | 6,649 |
| tabular+structural | 34 | 0.595 | 0.913 | 0.987 | 5,734 | 4,679 |
| tabular+structural+temporal | 42 | 0.662 | 0.921 | 1.000 | 5,036 | 3,981 |
| graph (all) | 62 | 0.911 | 0.986 | 1.000 | 1,137 | 82 |

Paired bootstrap PR-AUC(graph) - PR-AUC(tabular): 0.375 (95% CI 0.352 to 0.399, n=300)

## Top alerts

| rank | account_id | score | ground_truth_synthetic | detectors_fired | subgraph_nodes | subgraph_edges | motif_edges | top_reason | html |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24239 | 1.0 | scatter_gather | scatter_gather | 12 | 20 | 20 | motif_scatter_gather | alerts/alert_01_acct_24239.html |
| 2 | 30139 | 1.0 | scatter_gather | scatter_gather | 6 | 8 | 8 | motif_scatter_gather | alerts/alert_02_acct_30139.html |
| 3 | 29969 | 1.0 | fan_in | fan_in | 17 | 16 | 13 | motif_fan_in | alerts/alert_03_acct_29969.html |
| 4 | 72409 | 1.0 | mule_chain | mule_chain | 10 | 9 | 8 | motif_mule_chain | alerts/alert_04_acct_72409.html |
| 5 | 57647 | 1.0 | cycle | cycle | 8 | 8 | 7 | motif_cycle | alerts/alert_05_acct_57647.html |
| 6 | 18824 | 1.0 | fan_in | fan_in | 17 | 16 | 14 | motif_fan_in | alerts/alert_06_acct_18824.html |
| 7 | 82380 | 1.0 | scatter_gather | scatter_gather | 7 | 10 | 10 | motif_scatter_gather | alerts/alert_07_acct_82380.html |
| 8 | 22580 | 1.0 | cycle | cycle | 7 | 7 | 7 | motif_cycle | alerts/alert_08_acct_22580.html |
| 9 | 70098 | 1.0 | cycle | cycle | 6 | 6 | 4 | motif_cycle | alerts/alert_09_acct_70098.html |
| 10 | 18278 | 1.0 | fan_in | fan_in;fan_out | 58 | 63 | 62 | burstiness | alerts/alert_10_acct_18278.html |
