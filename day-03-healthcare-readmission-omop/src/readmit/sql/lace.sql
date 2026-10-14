-- LACE index (van Walraven et al., 2010): Length of stay, Acuity, Comorbidity (Charlson), ED visits (6 months).
CREATE OR REPLACE TABLE lace AS
SELECT
    visit_occurrence_id,
    CASE WHEN los_days < 1 THEN 0 WHEN los_days = 1 THEN 1 WHEN los_days = 2 THEN 2 WHEN los_days = 3 THEN 3
         WHEN los_days <= 6 THEN 4 WHEN los_days <= 13 THEN 5 ELSE 7 END AS lace_l,
    CASE WHEN emergent THEN 3 ELSE 0 END AS lace_a,
    CASE WHEN charlson >= 4 THEN 5 ELSE charlson END AS lace_c,
    least(prior_ed_180d, 4) AS lace_e
FROM features;
