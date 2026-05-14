"""
config.py
---------
Centralised path configuration for the leukemia survival project.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))

# Input files
CLINICAL_TRAIN = DATA_DIR / "X_train" / "clinical_train.csv"
MOLECULAR_TRAIN = DATA_DIR / "X_train" / "molecular_train.csv"
TARGET_TRAIN = DATA_DIR / "target_train.csv"
CLINICAL_TEST = DATA_DIR / "X_test" / "clinical_test.csv"
MOLECULAR_TEST = DATA_DIR / "X_test" / "molecular_test.csv"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

CLINICAL_MODEL_DIR = MODELS_DIR / "clinical"
CLINICAL_RSF = CLINICAL_MODEL_DIR / "rsf_model.joblib"
CLINICAL_PIPELINE = CLINICAL_MODEL_DIR / "preprocessing_pipeline.joblib"

FULL_MODEL_DIR = MODELS_DIR / "full"
FULL_RSF = FULL_MODEL_DIR / "rsf_model.joblib"
FULL_PIPELINE = FULL_MODEL_DIR / "preprocessing_pipeline.joblib"
FULL_SELECTED_FEATURES = FULL_MODEL_DIR / "selected_features.joblib"
FULL_MOLECULAR_FEATURES = FULL_MODEL_DIR / "molecular_features.joblib"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create dirs if needed
for d in [CLINICAL_MODEL_DIR, FULL_MODEL_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def check_data_files() -> None:
    """Verify that all expected input files exist."""
    files = {
        "clinical_train": CLINICAL_TRAIN,
        "molecular_train": MOLECULAR_TRAIN,
        "target_train": TARGET_TRAIN,
        "clinical_test": CLINICAL_TEST,
        "molecular_test": MOLECULAR_TEST,
    }
    print(f"DATA_DIR : {DATA_DIR}\n")
    all_ok = True
    for name, path in files.items():
        status = "✅" if path.exists() else "❌  NOT FOUND"
        print(f"  {status}  {name:25s}  {path}")
        if not path.exists():
            all_ok = False
    print("\nAll data files found." if all_ok else "\n⚠️  Some files are missing.")


def check_models() -> None:
    """Verify that trained models exist."""
    models = {
        "clinical/rsf_model": CLINICAL_RSF,
        "clinical/preprocessing_pipeline": CLINICAL_PIPELINE,
        "full/rsf_model": FULL_RSF,
        "full/preprocessing_pipeline": FULL_PIPELINE,
        "full/selected_features": FULL_SELECTED_FEATURES,
        "full/molecular_features": FULL_MOLECULAR_FEATURES,
    }
    print("Models status:\n")
    for name, path in models.items():
        status = "✅" if path.exists() else "❌  NOT FOUND"
        print(f"  {status}  {name}")
