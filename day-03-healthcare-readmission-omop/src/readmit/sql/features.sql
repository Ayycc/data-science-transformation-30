-- Discharge-time features for every eligible index admission. Everything is known at discharge:
-- history uses records dated before admission (utilisation) or up to discharge (conditions, drugs, labs).
CREATE OR REPLACE TABLE features AS
WITH idx AS (
    SELECT * FROM index_candidates WHERE exclusion_reason IS NULL
),
util AS (
    SELECT
        i.visit_occurrence_id,
        count(*) FILTER (WHERE v.visit_concept_id = 9201
                         AND v.visit_start_date >= i.admit_date - INTERVAL 365 DAY) AS prior_ip_365d,
        count(*) FILTER (WHERE v.visit_concept_id = 9203
                         AND v.visit_start_date >= i.admit_date - INTERVAL 180 DAY) AS prior_ed_180d,
        count(*) FILTER (WHERE v.visit_concept_id = 9202
                         AND v.visit_start_date >= i.admit_date - INTERVAL 365 DAY) AS prior_amb_365d
    FROM idx i
    LEFT JOIN visit_occurrence v
      ON v.person_id = i.person_id
     AND v.visit_start_date < i.admit_date
     AND v.visit_start_date >= i.admit_date - INTERVAL 365 DAY
    GROUP BY 1
),
comorb AS (
    SELECT
        i.visit_occurrence_id,
        coalesce(bool_or(c.condition_key = 'chf'), false) AS chf,
        coalesce(bool_or(c.condition_key = 'copd'), false) AS copd,
        coalesce(bool_or(c.condition_key = 'diabetes'), false) AS diabetes,
        coalesce(bool_or(c.condition_key = 'ckd'), false) AS ckd,
        coalesce(bool_or(c.condition_key = 'cancer'), false) AS cancer,
        coalesce(bool_or(c.condition_key = 'dementia'), false) AS dementia,
        coalesce(bool_or(c.condition_key = 'liver'), false) AS liver,
        coalesce(bool_or(c.condition_key = 'prior_mi'), false) AS prior_mi,
        coalesce(bool_or(c.condition_key = 'hypertension'), false) AS hypertension,
        coalesce(bool_or(c.condition_key = 'afib'), false) AS afib,
        coalesce(bool_or(c.condition_key = 'depression'), false) AS depression
    FROM idx i
    LEFT JOIN condition_occurrence c
      ON c.person_id = i.person_id
     AND c.condition_start_date <= i.discharge_date
    GROUP BY 1
),
drugs AS (
    SELECT i.visit_occurrence_id, count(DISTINCT d.drug_concept_id) AS n_drugs_365d
    FROM idx i
    LEFT JOIN drug_exposure d
      ON d.person_id = i.person_id
     AND d.drug_exposure_start_date BETWEEN i.discharge_date - INTERVAL 365 DAY AND i.discharge_date
    GROUP BY 1
),
labs AS (
    SELECT
        m.visit_occurrence_id,
        arg_max(m.value_as_number, m.measurement_datetime) FILTER (WHERE m.measurement_key = 'hemoglobin') AS hgb_last,
        min(m.value_as_number) FILTER (WHERE m.measurement_key = 'hemoglobin') AS hgb_min,
        arg_max(m.value_as_number, m.measurement_datetime) FILTER (WHERE m.measurement_key = 'sodium') AS sodium_last,
        arg_max(m.value_as_number, m.measurement_datetime) FILTER (WHERE m.measurement_key = 'creatinine') AS creat_last,
        max(m.value_as_number) FILTER (WHERE m.measurement_key = 'creatinine') AS creat_max,
        arg_max(m.value_as_number, m.measurement_datetime) FILTER (WHERE m.measurement_key = 'glucose') AS glucose_last,
        arg_max(m.value_as_number, m.measurement_datetime) FILTER (WHERE m.measurement_key = 'wbc') AS wbc_last
    FROM measurement m
    WHERE m.visit_occurrence_id IN (SELECT visit_occurrence_id FROM idx)
    GROUP BY 1
)
SELECT
    i.visit_occurrence_id,
    i.person_id,
    i.admit_date,
    i.discharge_date,
    i.age,
    CASE i.gender_concept_id WHEN 8507 THEN 'male' WHEN 8532 THEN 'female' ELSE 'unknown' END AS sex,
    date_diff('day', i.admit_date, i.discharge_date) AS los_days,
    (i.admit_source = 'emd') AS emergent,
    (coalesce(i.priority, '') = 'EL') AS elective,
    coalesce(i.primary_dx, 'unmapped') AS primary_dx,
    i.disposition,
    u.prior_ip_365d, u.prior_ed_180d, u.prior_amb_365d,
    c.chf, c.copd, c.diabetes, c.ckd, c.cancer, c.dementia, c.liver, c.prior_mi, c.hypertension, c.afib, c.depression,
    -- Charlson comorbidity index restricted to the conditions this vocabulary carries
    (c.chf::INT + c.copd::INT + c.diabetes::INT + 2 * c.ckd::INT + 2 * c.cancer::INT + c.dementia::INT
     + c.liver::INT + c.prior_mi::INT) AS charlson,
    dr.n_drugs_365d,
    l.hgb_last, l.hgb_min, l.sodium_last, l.creat_last, l.creat_max, l.glucose_last, l.wbc_last,
    i.readmit_30d,
    i.days_to_unplanned,
    i.n_planned_within_horizon
FROM idx i
JOIN util u USING (visit_occurrence_id)
JOIN comorb c USING (visit_occurrence_id)
JOIN drugs dr USING (visit_occurrence_id)
LEFT JOIN labs l USING (visit_occurrence_id)
ORDER BY i.visit_occurrence_id;
