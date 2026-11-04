# Release recommendation memo (generated from results/metrics.json)

**Data:** synthetic_generator (SYNTHETIC census-like population), 21,707 confidential records.
**Budget:** ε = 4, δ = 1e-05 for this file; spent ε = 3.5, δ = 1e-06.
**Risk gate:** no Anonymeter risk significantly above 0.05 (95% CI lower bound ≤ 0.05) and DCR membership-inference AUC ≤ 0.55. Among DP files that pass, pick the smallest ε whose TSTR AUC is within 0.005 of the best DP file.

## Recommendation

1. **Publish the four aggregate tables with Laplace noise (ε = 1 in total).** The median relative error of the race × sex × income counts is 1.43%, but Amer-Indian-Eskimo cells carry a mean relative error of 16.6% against 4.2% for the other groups: publish those cells with a flag and their noise scale, and point users of the rare group to coarser tables.
2. **Researcher microdata: release `DP tree (MST-style) eps=1`** (the formally DP file picked by the rule above). TSTR AUC 0.8302 vs 0.8747 on real data, max Anonymeter risk 0.046, membership AUC 0.5075.
3. `CTGAN-style GAN` scores higher (TSTR AUC 0.8570) and passes the empirical gate, but it has **no formal guarantee**: its risk numbers hold only for the attacks we ran. Use it only inside a secure research environment, not for open release.
4. **Do not release:** `k-anonymised microdata (k=5)` (max Anonymeter risk 0.328, membership AUC 0.691).
5. **Model release:** the ε = 1 logistic regression reaches AUC 0.8356 (non-private 0.8731); on the rare group AUC 0.8616 (non-private 0.8858).

## Caveats

- Empirical attack scores are lower bounds on risk; only the DP releases carry a worst-case guarantee.
- The rare subgroup has only 100 holdout records, so its AUCs have wide sampling error.
- The ε sweeps (curves, model and synthesiser grids) are what-if analyses on the confidential file; in production only the ledger entries above would be run.
- All numbers come from a SYNTHETIC population; the public UCI Adult file was unreachable from the build network.
