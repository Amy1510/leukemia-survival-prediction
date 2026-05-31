# Models

The trained models are not versioned in this repository due to their size (~3 GB total).

## Option 1 — Download pre-trained models

Contact the project maintainer to obtain the pre-trained models via a shared link.

Once downloaded, place the files as follows:

```
models/
├── clinical/
│   ├── rsf_model.joblib                # RSF trained on clinical features only
│   └── preprocessing_pipeline.joblib  # Fitted preprocessing pipeline
└── full/
    ├── rsf_model.joblib                # RSF trained on clinical + molecular features
    ├── preprocessing_pipeline.joblib  # Fitted preprocessing pipeline
    ├── selected_features.joblib        # List of feature column names
    └── molecular_features.joblib      # top_genes and top_effects metadata
```

## Option 2 — Retrain from scratch

Run the full training notebook:

```bash

poetry run jupyter notebook notebooks/survival_analysis.ipynb

```

The notebook will automatically save the models to `models/` upon completion.

## Model performance

| Model | CV C-index | Test platform |
|---|---|---|
| RSF — Clinical only | 0.7162 | 0.65 |
| **RSF — Clinical + Molecular** | **0.7393** | **0.7524** |

