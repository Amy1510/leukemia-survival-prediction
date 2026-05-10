"""
features.py
-----------
Feature engineering for AML survival prediction.

Covers:
- Cytogenetic classification (ISCN format → clinical risk groups)
- Categorical encoding utilities
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTINUOUS_VARS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
CATEGORICAL_VARS = ["CENTER", "CYTOGENETICS", "CYTO_CLASS"]
TARGET_TIME = "OS_YEARS"
TARGET_EVENT = "OS_STATUS"

CYTO_CLASS_ORDER = [
    "APL t(15;17)",  # best prognosis
    "inv(16)",
    "t(8;21)",
    "Normal",
    "Trisomy 8",
    "5q deletion",
    "Other",
    "Missing",
    "Monosomy 7",
    "Complex",          # worst prognosis
]


# ---------------------------------------------------------------------------
# Cytogenetic classification
# ---------------------------------------------------------------------------

def cytogenetic_group(karyotype: Optional[str]) -> str:
    """
    Map a raw ISCN karyotype string to a clinical risk group.

    Groups follow ELN 2022 AML classification standards:
    - Favorable: APL t(15;17), inv(16), t(8;21), Normal
    - Intermediate: Trisomy 8, 5q deletion, Other
    - Adverse: Monosomy 7, Complex (≥3 abnormalities)
    - Missing: no karyotype available

    Parameters
    ----------
    karyotype : str or None
        Raw ISCN string from the CYTOGENETICS column.

    Returns
    -------
    str
        Clinical risk group label.
    """
    if pd.isna(karyotype):
        return "Missing"

    x = karyotype.lower()

    # Favorable — recurrent AML translocations (check before Normal)
    if "t(15;17)" in x:
        return "APL t(15;17)"
    if "inv(16" in x:
        return "inv(16)"
    if "t(8;21)" in x:
        return "t(8;21)"

    # Normal karyotype: 46,XX or 46,XY without structural anomalies
    _normal_flags = ("del", "t(", "-7", "+8", "inv")
    if "46,xx" in x and not any(f in x for f in _normal_flags):
        return "Normal"
    if "46,xy" in x and not any(f in x for f in _normal_flags):
        return "Normal"

    # Adverse — monosomy / deletion
    if "-7" in x:
        return "Monosomy 7"
    if "del(5" in x or "5q" in x:
        return "5q deletion"
    if "+8" in x:
        return "Trisomy 8"

    # Complex karyotype: ≥3 abnormalities in the primary clone
    primary_clone = x.split("/")[0]
    if primary_clone.count(",") >= 3:
        return "Complex"

    return "Other"


def add_cyto_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a CYTO_CLASS column derived from CYTOGENETICS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a CYTOGENETICS column.

    Returns
    -------
    pd.DataFrame
        Copy of df with an added CYTO_CLASS column.
    """
    out = df.copy()
    out["CYTO_CLASS"] = out["CYTOGENETICS"].apply(cytogenetic_group)
    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_cyto_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode CYTO_CLASS according to clinical prognosis order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a CYTO_CLASS column.

    Returns
    -------
    pd.DataFrame
        Copy with an added CYTO_CLASS_ORD integer column (0 = best prognosis).
    """
    out = df.copy()
    order_map = {label: i for i, label in enumerate(CYTO_CLASS_ORDER)}
    out["CYTO_CLASS_ORD"] = out["CYTO_CLASS"].map(order_map)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    include_cyto: bool = True,
) -> pd.DataFrame:
    """
    Build the feature matrix used for modelling.

    Selects continuous hematological variables and, optionally, the
    ordinal-encoded cytogenetic class.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (outliers handled, CYTO_CLASS present).
    include_cyto : bool
        Whether to include CYTO_CLASS_ORD as a feature.

    Returns
    -------
    pd.DataFrame
        Feature matrix X.
    """
    cols = CONTINUOUS_VARS.copy()
    if include_cyto:
        df = encode_cyto_class(df)
        cols.append("CYTO_CLASS_ORD")
    return df[cols].copy()
