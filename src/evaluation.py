"""
evaluation.py
-------------
Evaluation utilities for AML survival models.

Covers:
- Concordance index (discrimination score) with confidence interval
- Kaplan-Meier survival curves (single group and stratified)
- Feature importance plot for Random Survival Forest
- Model comparison table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------

def concordance_index_ci(
    model,
    X: np.ndarray,
    y,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute concordance index with 95% bootstrap confidence interval.

    Parameters
    ----------
    model : fitted sksurv estimator (CoxPH or RSF)
    X : np.ndarray
    y : structured array (sksurv format)
    n_bootstrap : int
    random_state : int

    Returns
    -------
    dict with keys: c_index, ci_lower, ci_upper
    """
    from sksurv.metrics import concordance_index_censored

    rng = np.random.default_rng(random_state)
    n = len(y)

    # Point estimate
    risk_scores = model.predict(X)
    events = y[y.dtype.names[0]]
    times  = y[y.dtype.names[1]]
    c_index, _, _, _, _ = concordance_index_censored(events, times, risk_scores)

    # Bootstrap CI
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c, _, _, _, _ = concordance_index_censored(
                events[idx], times[idx], risk_scores[idx]
            )
            boot_scores.append(c)
        except Exception:
            continue

    boot_scores = np.array(boot_scores)
    return {
        "c_index":  round(c_index, 4),
        "ci_lower": round(float(np.percentile(boot_scores, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_scores, 97.5)), 4),
    }


def model_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a formatted comparison table from concordance_index_ci outputs.

    Parameters
    ----------
    results : dict  {model_name: {"c_index": ..., "ci_lower": ..., "ci_upper": ...}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Discrimination score": metrics["c_index"],
            "95% CI": f"[{metrics['ci_lower']} – {metrics['ci_upper']}]",
        })
    return pd.DataFrame(rows).sort_values("Discrimination score", ascending=False)


# ---------------------------------------------------------------------------
# Kaplan-Meier plots
# ---------------------------------------------------------------------------

_PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#65a30d",
]


def plot_kaplan_meier(
    df: pd.DataFrame,
    group_col: str,
    time_col: str = "OS_YEARS",
    event_col: str = "OS_STATUS",
    title: str = "",
    ax: Optional[plt.Axes] = None,
    ci_show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot stratified Kaplan-Meier curves with log-rank p-value.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str — stratification variable
    time_col : str
    event_col : str
    title : str
    ax : matplotlib Axes or None
    ci_show : bool — show confidence intervals
    save_path : str or None — path to save the figure

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    df_valid = df.dropna(subset=[time_col, event_col, group_col])
    kmf = KaplanMeierFitter()

    groups = sorted(df_valid[group_col].unique())
    for i, g in enumerate(groups):
        mask = df_valid[group_col] == g
        kmf.fit(
            durations=df_valid.loc[mask, time_col],
            event_observed=df_valid.loc[mask, event_col],
            label=str(g),
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=ci_show,
            color=_PALETTE[i % len(_PALETTE)],
            show_censors=True,
            censor_styles={"ms": 4, "marker": "|"},
        )

    # Log-rank p-value
    res = multivariate_logrank_test(
        df_valid[time_col], df_valid[group_col], df_valid[event_col]
    )
    pval = res.p_value
    pval_str = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"

    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(f"{title}\nLog-rank {pval_str}", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Feature importance (RSF)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 10,
    title: str = "Feature Importance — Random Survival Forest",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Horizontal bar chart of RSF permutation feature importances.

    Parameters
    ----------
    model : fitted RandomSurvivalForest
    feature_names : list of str
    top_n : int
    title : str
    ax : matplotlib Axes or None
    save_path : str or None

    Returns
    -------
    matplotlib Axes
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.4)))

    colors = ["#2563eb" if v == vals.max() else "#93c5fd" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="none")
    ax.set_xlabel("Mean decrease in impurity", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Predicted survival curve for a single patient (used by Streamlit app)
# ---------------------------------------------------------------------------

def predict_survival_curve(
    model,
    X_patient: np.ndarray,
    label: str = "Patient",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot the predicted survival function for a single patient.

    Parameters
    ----------
    model : fitted RSF or CoxPH (sksurv)
    X_patient : np.ndarray, shape (1, n_features)
    label : str
    ax : matplotlib Axes or None
    save_path : str or None

    Returns
    -------
    matplotlib Axes
    """
    surv_fn = model.predict_survival_function(X_patient)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    fn = surv_fn[0]
    ax.step(fn.x, fn(fn.x), where="post", color="#2563eb", lw=2, label=label)
    ax.fill_between(fn.x, fn(fn.x), alpha=0.12, color="#2563eb", step="post")

    # Reference lines
    for prob, style in [(0.5, "--"), (0.25, ":")]:
        ax.axhline(prob, color="gray", ls=style, lw=0.8, alpha=0.6)

    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(f"Predicted survival curve — {label}", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
