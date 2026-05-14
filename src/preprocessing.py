"""
preprocessing.py
----------------
Data cleaning and preprocessing utilities for AML survival prediction.

Covers:
- IQR-based univariate outlier detection
- Isolation Forest multivariate outlier detection
- Missing value analysis
- sklearn-compatible preprocessing pipeline (no data leakage)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.features import CONTINUOUS_VARS, TARGET_TIME, TARGET_EVENT

# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def iqr_outlier_bounds(
    series: pd.Series,
    factor: float = 1.5,
) -> Dict[str, float]:
    """
    Compute IQR-based outlier bounds for a numeric series.

    Parameters
    ----------
    series : pd.Series
    factor : float
        IQR multiplier (1.5 = standard, 3.0 = extreme outliers only).

    Returns
    -------
    dict with keys: q1, q3, iqr, lower, upper
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": q1 - factor * iqr,
        "upper": q3 + factor * iqr,
    }


def detect_iqr_outliers(
    df: pd.DataFrame,
    feature: str,
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Detect univariate outliers using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
    factor : float

    Returns
    -------
    outliers : pd.DataFrame — rows flagged as outliers
    bounds : dict — q1, q3, iqr, lower, upper
    """
    bounds = iqr_outlier_bounds(df[feature].dropna(), factor)
    mask = (df[feature] < bounds["lower"]) | (df[feature] > bounds["upper"])
    return df[mask], bounds


def summarize_outliers(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Return a summary table of IQR outliers for multiple features.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str (defaults to CONTINUOUS_VARS)
    factor : float

    Returns
    -------
    pd.DataFrame with columns: feature, n_outliers, lower, upper, pct
    """
    if features is None:
        features = CONTINUOUS_VARS
    rows = []
    for feat in features:
        outliers, bounds = detect_iqr_outliers(df, feat, factor)
        rows.append(
            {
                "feature": feat,
                "n_outliers": len(outliers),
                "pct_outliers": round(100 * len(outliers) / len(df), 2),
                "lower_bound": round(bounds["lower"], 3),
                "upper_bound": round(bounds["upper"], 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Missing value analysis
# ---------------------------------------------------------------------------


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing values per column.
    """
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    return (
        pd.DataFrame({"n_missing": missing, "pct_missing": pct.round(2)})
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )


# ---------------------------------------------------------------------------
# Preprocessing pipeline (leak-free)
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline for continuous hematological features.

    Steps:
    1. Median imputation — robust to clinical outliers
    2. Standard scaling — required by Cox PH

    The pipeline is fit only on training data to avoid leakage.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_survival_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare (X, y) for scikit-survival models.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature_cols, TARGET_TIME, TARGET_EVENT.
    feature_cols : list of str
    pipeline : fitted Pipeline or None
        If None and fit_pipeline=True, a new pipeline is built and fitted.
    fit_pipeline : bool
        If True, fit the pipeline on df. If False, only transform.

    Returns
    -------
    X : np.ndarray, shape (n, p)
    y : structured array with fields ('OS_STATUS', bool) and ('OS_YEARS', float)
    pipeline : fitted Pipeline
    """
    from sksurv.util import Surv

    df_clean = df.dropna(subset=feature_cols + [TARGET_TIME, TARGET_EVENT]).copy()

    X_raw = df_clean[feature_cols].values

    if pipeline is None:
        pipeline = build_preprocessing_pipeline()

    if fit_pipeline:
        X = pipeline.fit_transform(X_raw)
    else:
        X = pipeline.transform(X_raw)

    y = Surv.from_dataframe(TARGET_EVENT, TARGET_TIME, df_clean)

    return X, y, pipeline


# Normalisation
def log1p_cols(X):
    """Log1p transformation for skewed clinical variables."""
    X = X.copy()
    for c in ["WBC", "ANC", "MONOCYTES", "PLT"]:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    return X


def log1p_full(X):
    """Log1p transformation for skewed clinical + molecular variables."""
    X = X.copy()
    for c in [
        "WBC",
        "ANC",
        "MONOCYTES",
        "PLT",
        "N_MUT",
        "N_GENES",
        "DEPTH_MAX",
        "DEPTH_MEAN",
    ]:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    return X
