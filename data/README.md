# Data

## Source

This project uses data from the **QRT Data Challenge 2025** — Overall Survival prediction for Acute Myeloid Leukemia (AML).

Challenge page: [challengedata.ens.fr/participants/challenges/167/](https://challengedata.ens.fr/participants/challenges/167/)

The dataset was provided by QRT in collaboration with European hospital centres. It contains
clinical and molecular data from **3,323 AML patients** across 10 centres.

---

## Why the data is not included

The dataset is subject to the QRT Data Challenge terms of use and is not redistributable.
It contains sensitive patient data (even if anonymised) and must be accessed through the
official challenge platform only.

---

## How to access the data

1. Create an account on [challengedata.ens.fr](https://challengedata.ens.fr)
2. Navigate to challenge **#167** — QRT 2025 AML Survival
3. Accept the terms of use
4. Download the dataset files

---

## Expected structure

Once downloaded, place the files in this `data/` folder as follows:

```
data/
├── X_train/
│   ├── clinical_train.csv      # Clinical features — one row per patient (train)
│   └── molecular_train.csv     # Molecular data — one row per mutation (train)
├── X_test/
│   ├── clinical_test.csv       # Clinical features — one row per patient (test)
│   └── molecular_test.csv      # Molecular data — one row per mutation (test)
└── target_train.csv            # Survival target — OS_YEARS and OS_STATUS (train)
```

---

## Dataset description

### clinical_train.csv / clinical_test.csv

| Column | Type | Description |
|---|---|---|
| ID | int | Patient identifier |
| CENTER | str | Hospital centre (anonymised) |
| CYTOGENETICS | str | Karyotype in ISCN format |
| BM_BLAST | float | Bone marrow blast percentage (%) |
| WBC | float | White blood cell count (G/L) |
| ANC | float | Absolute neutrophil count (G/L) |
| MONOCYTES | float | Monocyte count (G/L) |
| HB | float | Haemoglobin level (g/dL) |
| PLT | float | Platelet count (G/L) |

### target_train.csv

| Column | Type | Description |
|---|---|---|
| ID | int | Patient identifier |
| OS_YEARS | float | Overall survival time (years) |
| OS_STATUS | int | Event indicator — 1 = death observed, 0 = censored |

### molecular_train.csv / molecular_test.csv

One row per mutation. Key columns: GENE, EFFECT, VAF (variant allele frequency), DEPTH (sequencing depth).
Each patient may have multiple rows (multiple mutations).