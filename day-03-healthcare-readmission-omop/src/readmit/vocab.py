"""Small code vocabulary used by the generator and the normaliser.

Standard OMOP concept ids are used only for gender (8507/8532), visit (9201/9202/9203) and
type (32817, "EHR"). Clinical concepts get *local* concept ids in the OMOP-reserved range
(>= 2,000,000,000); the original source code is kept in ``*_source_value``. No OHDSI Athena
vocabulary is shipped. ICD-10-CM, SNOMED CT, RxNorm and LOINC codes below are used as labels
on SYNTHETIC data and are mapped by hand.
"""

from __future__ import annotations

GENDER_CONCEPT = {"male": 8507, "female": 8532}  # other/unknown -> 0
VISIT_CONCEPT = {"IMP": 9201, "AMB": 9202, "EMER": 9203}
EHR_TYPE_CONCEPT = 32817

ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC = "http://loinc.org"

# key: (local concept id, ICD-10-CM code, SNOMED code, display, Charlson weight, chronic?)
CONDITIONS: dict[str, tuple[int, str, str, str, int, bool]] = {
    "chf": (2_000_000_001, "I50.9", "42343007", "Congestive heart failure", 1, True),
    "copd": (2_000_000_002, "J44.9", "13645005", "Chronic obstructive pulmonary disease", 1, True),
    "diabetes": (2_000_000_003, "E11.9", "44054006", "Type 2 diabetes mellitus", 1, True),
    "ckd": (2_000_000_004, "N18.4", "431857002", "Chronic kidney disease stage 4", 2, True),
    "cancer": (2_000_000_005, "C34.90", "254637007", "Malignant neoplasm of lung", 2, True),
    "dementia": (2_000_000_006, "F03.90", "52448006", "Dementia", 1, True),
    "liver": (2_000_000_007, "K70.30", "420054005", "Alcoholic cirrhosis of liver", 1, True),
    "prior_mi": (2_000_000_008, "I25.2", "1755008", "Old myocardial infarction", 1, True),
    "hypertension": (2_000_000_009, "I10", "38341003", "Essential hypertension", 0, True),
    "afib": (2_000_000_010, "I48.91", "49436004", "Atrial fibrillation", 0, True),
    "depression": (2_000_000_011, "F32.9", "35489007", "Depressive disorder", 0, True),
    # acute admission reasons
    "pneumonia": (2_000_000_101, "J18.9", "233604007", "Pneumonia", 0, False),
    "sepsis": (2_000_000_102, "A41.9", "91302008", "Sepsis", 0, False),
    "uti": (2_000_000_103, "N39.0", "68566005", "Urinary tract infection", 0, False),
    "ami": (2_000_000_104, "I21.9", "22298006", "Acute myocardial infarction", 0, False),
    "gi_bleed": (2_000_000_105, "K92.2", "74474003", "Gastrointestinal hemorrhage", 0, False),
    "hip_fracture": (2_000_000_106, "S72.009A", "5913000", "Fracture of neck of femur", 0, False),
    "chf_exac": (2_000_000_107, "I50.23", "48447003", "Acute on chronic heart failure", 0, False),
    "copd_exac": (2_000_000_108, "J44.1", "195951007", "Acute exacerbation of COPD", 0, False),
    "chemo": (2_000_000_109, "Z51.11", "367336001", "Encounter for antineoplastic chemotherapy", 0, False),
    "elective_joint": (2_000_000_110, "Z47.1", "609588000", "Elective joint replacement aftercare", 0, False),
    "cad_staged": (2_000_000_111, "I25.10", "53741008", "Coronary atherosclerosis (staged PCI)", 0, False),
}
CHRONIC = [k for k, v in CONDITIONS.items() if v[5]]
ICD10_TO_KEY = {v[1]: k for k, v in CONDITIONS.items()}
SNOMED_TO_KEY = {v[2]: k for k, v in CONDITIONS.items()}

# key: (local concept id, RxNorm ingredient code, display)
DRUGS: dict[str, tuple[int, str, str]] = {
    "furosemide": (2_000_001_001, "4603", "furosemide"),
    "lisinopril": (2_000_001_002, "29046", "lisinopril"),
    "metoprolol": (2_000_001_003, "6918", "metoprolol"),
    "metformin": (2_000_001_004, "6809", "metformin"),
    "insulin_glargine": (2_000_001_005, "274783", "insulin glargine"),
    "tiotropium": (2_000_001_006, "69120", "tiotropium"),
    "albuterol": (2_000_001_007, "435", "albuterol"),
    "apixaban": (2_000_001_008, "1364430", "apixaban"),
    "atorvastatin": (2_000_001_009, "83367", "atorvastatin"),
    "aspirin": (2_000_001_010, "1191", "aspirin"),
    "sertraline": (2_000_001_011, "36437", "sertraline"),
    "donepezil": (2_000_001_012, "135447", "donepezil"),
    "amlodipine": (2_000_001_013, "17767", "amlodipine"),
    "ceftriaxone": (2_000_001_014, "2193", "ceftriaxone"),
    "prednisone": (2_000_001_015, "8640", "prednisone"),
    "spironolactone": (2_000_001_016, "9997", "spironolactone"),
    "pantoprazole": (2_000_001_017, "40790", "pantoprazole"),
    "oxycodone": (2_000_001_018, "7804", "oxycodone"),
    "cisplatin": (2_000_001_019, "2555", "cisplatin"),
    "lactulose": (2_000_001_020, "6218", "lactulose"),
}
RXNORM_TO_KEY = {v[1]: k for k, v in DRUGS.items()}

# Chronic condition -> maintenance drugs
CHRONIC_DRUGS = {
    "chf": ["furosemide", "metoprolol", "spironolactone", "lisinopril"],
    "copd": ["tiotropium", "albuterol"],
    "diabetes": ["metformin", "insulin_glargine", "atorvastatin"],
    "ckd": ["amlodipine"],
    "cancer": ["oxycodone"],
    "dementia": ["donepezil"],
    "liver": ["lactulose"],
    "prior_mi": ["aspirin", "atorvastatin", "metoprolol"],
    "hypertension": ["lisinopril", "amlodipine"],
    "afib": ["apixaban", "metoprolol"],
    "depression": ["sertraline"],
}

# Labs: key -> (local concept id, LOINC, display, canonical unit, {unit: factor to canonical})
LABS: dict[str, tuple[int, str, str, str, dict[str, float]]] = {
    "hemoglobin": (2_000_002_001, "718-7", "Hemoglobin", "g/dL", {"g/dL": 1.0, "g/L": 0.1}),
    "sodium": (2_000_002_002, "2951-2", "Sodium", "mmol/L", {"mmol/L": 1.0, "mEq/L": 1.0}),
    "creatinine": (2_000_002_003, "2160-0", "Creatinine", "mg/dL", {"mg/dL": 1.0, "umol/L": 1 / 88.42}),
    "glucose": (2_000_002_004, "2345-7", "Glucose", "mg/dL", {"mg/dL": 1.0, "mmol/L": 18.016}),
    "wbc": (2_000_002_005, "6690-2", "Leukocytes", "10*3/uL", {"10*3/uL": 1.0, "10*9/L": 1.0}),
    "systolic_bp": (2_000_002_006, "8480-6", "Systolic blood pressure", "mm[Hg]", {"mm[Hg]": 1.0}),
}
LOINC_TO_KEY = {v[1]: k for k, v in LABS.items()}
# Plausible canonical ranges; values outside are rejected by the schema (likely unit or entry error).
LAB_PLAUSIBLE = {
    "hemoglobin": (2.0, 25.0),
    "sodium": (95.0, 180.0),
    "creatinine": (0.1, 25.0),
    "glucose": (10.0, 1500.0),
    "wbc": (0.1, 200.0),
    "systolic_bp": (40.0, 300.0),
}

# HL7 discharge-disposition codes -> local concept ids. "other-hcf" is used for transfers
# to another acute-care hospital in this synthetic data.
DISCHARGE = {
    "home": 2_000_003_001,
    "snf": 2_000_003_002,
    "rehab": 2_000_003_003,
    "hosp": 2_000_003_004,  # hospice
    "long": 2_000_003_005,
    "other-hcf": 2_000_003_006,
    "aadvice": 2_000_003_007,
    "exp": 2_000_003_008,
}
ADMIT_SOURCE = {"emd": 2_000_004_001, "gp": 2_000_004_002, "outp": 2_000_004_003, "hosp-trans": 2_000_004_004}
PRIORITY = {"EM", "UR", "EL", "R"}  # v3 ActPriority subset: emergency, urgent, elective, routine
