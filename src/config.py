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
FULL_MODEL_DIR = MODELS_DIR / "full"

# Raw models straight out of the training notebook (survival_analysis.ipynb).
# These are big (1-2 GB) and NOT tracked in git, the notebook should only
# ever read/write through these two paths, never the compressed ones below.
CLINICAL_RSF_RAW = CLINICAL_MODEL_DIR / "rsf_model.joblib"
FULL_RSF_RAW = FULL_MODEL_DIR / "rsf_model.joblib"

# What the deployed app actually loads. Compressed + pruned version of the
# raw models above, produced by scripts/compress_models.py. Small enough to
# live in the repo and to fit in Streamlit Cloud's memory limits.
CLINICAL_RSF = CLINICAL_MODEL_DIR / "rsf_model_compressed.joblib"
FULL_RSF = FULL_MODEL_DIR / "rsf_model_compressed.joblib"

CLINICAL_PIPELINE = CLINICAL_MODEL_DIR / "preprocessing_pipeline.joblib"
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
    """Verify that trained (raw) and deployed (compressed) models exist."""
    models = {
        "clinical/rsf_model (raw, straight from training)": CLINICAL_RSF_RAW,
        "clinical/rsf_model_compressed (what the app uses)": CLINICAL_RSF,
        "clinical/preprocessing_pipeline": CLINICAL_PIPELINE,
        "full/rsf_model (raw, straight from training)": FULL_RSF_RAW,
        "full/rsf_model_compressed (what the app uses)": FULL_RSF,
        "full/preprocessing_pipeline": FULL_PIPELINE,
        "full/selected_features": FULL_SELECTED_FEATURES,
        "full/molecular_features": FULL_MOLECULAR_FEATURES,
    }
    print("Models status:\n")
    for name, path in models.items():
        status = "✅" if path.exists() else "❌  NOT FOUND"
        print(f"  {status}  {name}")
