-- Index admissions for 30-day all-cause unplanned readmission (CMS-style, simplified).
-- Parameters: $lookback_start, $followup_end, $horizon (days).
-- Every inpatient discharge gets exactly one exclusion_reason (the first rule it fails, in the
-- order below) or NULL if it is an eligible index admission. The attrition table is a GROUP BY.
CREATE OR REPLACE TABLE index_candidates AS
WITH ip AS (
    SELECT v.*, p.birth_datetime, p.gender_concept_id
    FROM visit_occurrence v
    JOIN person p USING (person_id)
    WHERE v.visit_concept_id = 9201  -- inpatient visit
),
primary_dx AS (
    SELECT visit_occurrence_id, arg_min(condition_key, condition_occurrence_id) AS primary_dx
    FROM condition_occurrence
    WHERE visit_occurrence_id IS NOT NULL AND condition_status_source_value = 'primary'
    GROUP BY 1
),
followups AS (
    -- inpatient admissions starting on or after the index discharge date, within the horizon
    SELECT
        i.visit_occurrence_id,
        min(CASE WHEN coalesce(r.priority_source_value, '') <> 'EL'
                 THEN date_diff('day', i.visit_end_date, r.visit_start_date) END) AS days_to_unplanned,
        count(*) FILTER (WHERE r.priority_source_value = 'EL') AS n_planned_within_horizon
    FROM ip i
    JOIN ip r
      ON r.person_id = i.person_id
     AND r.visit_occurrence_id <> i.visit_occurrence_id
     AND r.visit_start_date >= i.visit_end_date
     AND r.visit_start_date <= i.visit_end_date + to_days(CAST($horizon AS INTEGER))
    GROUP BY 1
)
SELECT
    ip.visit_occurrence_id,
    ip.person_id,
    ip.visit_start_date AS admit_date,
    ip.visit_end_date AS discharge_date,
    ip.gender_concept_id,
    date_diff('day', ip.birth_datetime::DATE, ip.visit_start_date) / 365.25 AS age,
    ip.discharged_to_source_value AS disposition,
    ip.admitted_from_source_value AS admit_source,
    ip.priority_source_value AS priority,
    dx.primary_dx,
    d.death_date,
    f.days_to_unplanned,
    coalesce(f.n_planned_within_horizon, 0) AS n_planned_within_horizon,
    CASE
        WHEN ip.visit_end_date < $lookback_start THEN '01 discharge inside 365-day lookback'
        WHEN ip.visit_end_date > $followup_end THEN '02 under 30 days of follow-up left'
        WHEN date_diff('day', ip.birth_datetime::DATE, ip.visit_start_date) / 365.25 < 18 THEN '03 age under 18'
        WHEN ip.discharged_to_source_value = 'exp' OR d.death_date <= ip.visit_end_date THEN '04 died in hospital'
        WHEN ip.discharged_to_source_value = 'other-hcf' THEN '05 transferred to another acute hospital'
        WHEN ip.discharged_to_source_value = 'aadvice' THEN '06 left against medical advice'
        WHEN dx.primary_dx = 'chemo' THEN '07 planned admission for chemotherapy'
        WHEN d.death_date <= ip.visit_end_date + to_days(CAST($horizon AS INTEGER))
             AND (f.days_to_unplanned IS NULL
                  OR ip.visit_end_date + to_days(CAST(f.days_to_unplanned AS INTEGER)) > d.death_date)
            THEN '08 died within 30 days before any readmission'
    END AS exclusion_reason,
    (f.days_to_unplanned IS NOT NULL) AS readmit_30d
FROM ip
LEFT JOIN primary_dx dx USING (visit_occurrence_id)
LEFT JOIN death d ON d.person_id = ip.person_id
LEFT JOIN followups f ON f.visit_occurrence_id = ip.visit_occurrence_id
ORDER BY ip.visit_occurrence_id;
