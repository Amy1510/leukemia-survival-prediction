"""
app.py, Leukemia Survival Predictor (bilingual EN / FR)
Usage: streamlit run app/app.py

"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

from matplotlib.patches import Patch
from src.config import CLINICAL_RSF, FULL_RSF
from src.preprocessing import log1p_cols, log1p_full

from i18n import (
    t as _t,
    LANGUAGES,
    CYTO_DESC,
    CYTO_DISPLAY,
    RANGE_HINTS,
    FEATURE_LABELS_I18N,
    MOL_PROFILES_I18N,
    FAMILIES,
    BURDEN_LABELS,
)

st.set_page_config(
    page_title="Leukemia Survival Predictor",
    page_icon="🩺",
    layout="wide",
)


# Language selector — rendered at the top of the sidebar
LANG = st.sidebar.selectbox(
    "🌐 Language / Langue",
    options=list(LANGUAGES.keys()),
    format_func=lambda k: LANGUAGES[k],
    index=0,
    key="lang",
)
st.sidebar.markdown("---")


def T(key, **kw):
    """Shorthand: translate `key` in the currently selected language."""
    return _t(key, LANG, **kw)


# Constants: clinical

CYTO_CLASSES = [
    "APL t(15;17)",
    "inv(16)",
    "t(8;21)",
    "Normal",
    "Trisomy 8",
    "5q deletion",
    "Monosomy 7",
    "Complex (≥3 anomalies)",
]

CYTO_LABELS_MAP = {"Complex (≥3 anomalies)": "Complex"}

NORMAL_RANGES = {
    "BM_BLAST": (0, 5),
    "WBC": (4, 10),
    "ANC": (1.8, 7),
    "MONOCYTES": (0.2, 1),
    "HB": (12, 16),
    "PLT": (150, 400),
}

CENTER_VALUE = "KI"  # placeholder center, consistent with training data encoding


# Constants — molecular

MOLECULAR_GENE_COLS = [
    "TET2",
    "ASXL1",
    "SF3B1",
    "DNMT3A",
    "RUNX1",
    "SRSF2",
    "TP53",
    "STAG2",
    "U2AF1",
    "EZH2",
    "CBL",
    "BCOR",
    "NRAS",
    "ZRSR2",
    "DDX41",
    "IDH2",
    "CUX1",
    "NF1",
    "PHF6",
    "KRAS",
    "SETBP1",
    "JAK2",
    "MLL",
    "PTPN11",
    "CEBPA",
    "IDH1",
    "ETV6",
    "ETNK1",
    "MPL",
    "SH2B3",
]

MOLECULAR_EFFECT_COLS = [
    "2KB_upstream_variant",
    "ITD",
    "OTHER_EFFECT",
    "PTD",
    "frameshift_variant",
    "inframe_codon_gain",
    "inframe_codon_loss",
    "initiator_codon_change",
    "non_synonymous_codon",
    "splice_site_variant",
    "stop_gained",
]

# Illustrative synthetic profiles, NOT real patient data.
MOLECULAR_PROFILES = {
    "none": {
        "genes": [],
        "effects": [],
        "n_mut": 0,
        "n_genes": 0,
        "vaf_mean": 0.0,
        "vaf_max": 0.0,
        "depth_mean": 0.0,
        "depth_max": 0.0,
        "has_mol_data": 0,
    },
    "favorable": {
        "genes": ["CEBPA"],
        "effects": ["frameshift_variant"],
        "n_mut": 1,
        "n_genes": 1,
        "vaf_mean": 0.42,
        "vaf_max": 0.42,
        "depth_mean": 280.0,
        "depth_max": 280.0,
        "has_mol_data": 1,
    },
    "adverse": {
        "genes": ["TP53"],
        "effects": ["non_synonymous_codon"],
        "n_mut": 2,
        "n_genes": 1,
        "vaf_mean": 0.58,
        "vaf_max": 0.65,
        "depth_mean": 240.0,
        "depth_max": 260.0,
        "has_mol_data": 1,
    },
}

CINDEX_CLINICAL = 0.65
CINDEX_FULL = 0.752


# Localised lookups (depend on LANG)

FEATURE_LABELS = FEATURE_LABELS_I18N[LANG]
_RANGE_HINT = RANGE_HINTS[LANG]
_MOL_TEXT = MOL_PROFILES_I18N[LANG]
_FAM = FAMILIES[LANG]
_BURDEN = BURDEN_LABELS[LANG]


def mol_label(key):
    return _MOL_TEXT[key]["label"]


def mol_description(key):
    return _MOL_TEXT[key]["description"]


# Model loading


@st.cache_resource(show_spinner=False)
def load_clinical_model():
    return joblib.load(CLINICAL_RSF)


@st.cache_resource(show_spinner=False)
def load_full_model():
    return joblib.load(FULL_RSF)


clinical_model, clinical_loaded = None, False
full_model, full_loaded = None, False
load_error = ""

try:
    clinical_model = load_clinical_model()
    clinical_loaded = True
except FileNotFoundError as e:
    load_error = str(e)

try:
    full_model = load_full_model()
    full_loaded = True
except FileNotFoundError as e:
    load_error = load_error or str(e)


# SHAP explainability (memory-safe: on demand + cached)

SHAP_BG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "full", "shap_background.parquet"
)


@st.cache_data(show_spinner=False)
def load_shap_background():
    return pd.read_parquet(SHAP_BG_PATH)


def feature_family(name):
    if name == "CYTO_CLASS":
        return _FAM["cyto"]
    if name == "CENTER":
        return _FAM["center"]
    if name in ("BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"):
        return _FAM["hema"]
    if name in (
        "N_MUT",
        "N_GENES",
        "VAF_MEAN",
        "VAF_MAX",
        "DEPTH_MEAN",
        "DEPTH_MAX",
        "HAS_MOL_DATA",
    ):
        return _FAM["burden"]
    if name.startswith("GENE_"):
        return _FAM["genes"]
    if name.startswith("EFFECT_"):
        return _FAM["effects"]
    return name


def pretty_label(name):
    """Readable label for a raw feature (e.g. GENE_TP53 -> 'TP53 mutation')."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name == "CYTO_CLASS":
        return T("lab_cyto")
    if name == "CENTER":
        return T("lab_center")
    if name.startswith("GENE_"):
        return T("lab_mutation", g=name[5:])
    if name.startswith("EFFECT_"):
        return T("lab_effect", e=name[7:].replace("_", " "))
    return _BURDEN.get(name, name)


def _surv_at_t_batched(model, X_raw, t_star, batch=200):
    """S(t*) in small batches, freeing memory in between."""
    out = []
    for i in range(0, len(X_raw), batch):
        fns = model.predict_survival_function(X_raw.iloc[i : i + batch])
        out.append(np.array([float(fn(t_star)) for fn in fns]))
        del fns
        gc.collect()
    return np.concatenate(out)


CLINICAL_COLS = [
    "BM_BLAST",
    "WBC",
    "ANC",
    "MONOCYTES",
    "HB",
    "PLT",
    "CYTO_CLASS",
    "CENTER",
]


@st.cache_data(show_spinner=False)
def compute_patient_shap(row_items, t_star, background_n, nsamples, model_kind="full"):
    """SHAP for one patient at horizon t*, on the same model shown above."""

    bg_full = load_shap_background()

    if model_kind == "clinical":
        model = clinical_model
        cols = [c for c in CLINICAL_COLS if c in bg_full.columns]
    else:
        model = full_model
        cols = list(bg_full.columns)

    bg = bg_full[cols]
    num_cols = list(bg.select_dtypes(include="number").columns)
    patient = pd.DataFrame([dict(row_items)]).reindex(columns=cols)

    times = model.predict_survival_function(patient)[0].x
    t_star = float(min(t_star, float(np.max(times))))

    bg_s = shap.sample(bg, min(background_n, len(bg)), random_state=0)

    def f(data):
        df = pd.DataFrame(data, columns=cols)
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        return _surv_at_t_batched(model, df, t_star)

    explainer = shap.KernelExplainer(f, bg_s)
    sv = np.array(
        explainer.shap_values(patient.values, nsamples=nsamples, silent=True)
    ).reshape(-1)
    base = float(explainer.expected_value)
    s_patient = float(_surv_at_t_batched(model, patient, t_star)[0])
    gc.collect()
    return {
        "shap": sv,
        "cols": cols,
        "base": base,
        "s_patient": s_patient,
        "t_star": t_star,
    }


# Feature construction


def build_clinical_row(bm_blast, wbc, anc, monocytes, hb, plt_count, cyto_class):
    return {
        "BM_BLAST": bm_blast,
        "WBC": wbc,
        "ANC": anc,
        "MONOCYTES": monocytes,
        "HB": hb,
        "PLT": plt_count,
        "CYTO_CLASS": cyto_class,
        "CENTER": CENTER_VALUE,
    }


def build_molecular_row(profile_key: str) -> dict:
    profile = MOLECULAR_PROFILES[profile_key]
    row = {f"GENE_{g}": 0 for g in MOLECULAR_GENE_COLS}
    row.update({f"EFFECT_{e}": 0 for e in MOLECULAR_EFFECT_COLS})
    for gene in profile["genes"]:
        row[f"GENE_{gene}"] = 1
    for effect in profile["effects"]:
        row[f"EFFECT_{effect}"] = 1
    row["N_MUT"] = profile["n_mut"]
    row["N_GENES"] = profile["n_genes"]
    row["VAF_MEAN"] = profile["vaf_mean"]
    row["VAF_MAX"] = profile["vaf_max"]
    row["DEPTH_MEAN"] = profile["depth_mean"]
    row["DEPTH_MAX"] = profile["depth_max"]
    row["HAS_MOL_DATA"] = profile["has_mol_data"]
    return row


# Welcome message

st.title(T("app_title"))

st.markdown(
    f"""
<div style="background-color:#1e3a5f; padding:16px; border-radius:8px; margin-bottom:16px; color:#eef2fa !important;">
<b style="color:#eef2fa;">{T("intro_what")}</b><br>
<span style="color:#eef2fa;">{T("intro_body")}</span><br><br>
<b style="color:#eef2fa;">{T("intro_who")}</b> <span style="color:#eef2fa;">{T("intro_who_body")}</span><br>
<b style="color:#eef2fa;">{T("intro_warn")}</b> <span style="color:#eef2fa;">{T("intro_warn_body")}</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander(T("howto_title"), expanded=False):
    st.markdown(T("howto_body"))

st.divider()

if not (clinical_loaded and full_loaded):
    st.error(T("err_nomodel", err=load_error))
    st.stop()


# Sidebar - clinical inputs

st.sidebar.header(T("sb_header"))

st.sidebar.markdown(T("sb_cyto"))
cyto_display = st.sidebar.selectbox(
    T("sb_caryotype"),
    CYTO_CLASSES,
    index=CYTO_CLASSES.index("Normal"),
    format_func=lambda k: CYTO_DISPLAY[LANG].get(k, k),
    label_visibility="collapsed",
)
cyto_class = CYTO_LABELS_MAP.get(cyto_display, cyto_display)
st.sidebar.caption(f"ℹ️ {CYTO_DESC[LANG][cyto_display]}")

st.sidebar.markdown("---")
st.sidebar.markdown(T("sb_hema"))

col_a, col_b = st.sidebar.columns(2)

with col_a:
    st.markdown("<small>BM_BLAST (%)</small>", unsafe_allow_html=True)
    bm_blast = st.number_input(
        "BM_BLAST", 0.0, 100.0, 5.0, 0.5, label_visibility="collapsed", key="bm"
    )
    st.caption(_RANGE_HINT["BM_BLAST"])

    st.markdown("<small>ANC (G/L)</small>", unsafe_allow_html=True)
    anc = st.number_input(
        "ANC", 0.0, 120.0, 3.0, 0.1, label_visibility="collapsed", key="anc"
    )
    st.caption(_RANGE_HINT["ANC"])

    st.markdown("<small>HB (g/dL)</small>", unsafe_allow_html=True)
    hb = st.number_input(
        "HB", 4.0, 18.0, 10.0, 0.1, label_visibility="collapsed", key="hb"
    )
    st.caption(_RANGE_HINT["HB"])

with col_b:
    st.markdown("<small>WBC (G/L)</small>", unsafe_allow_html=True)
    wbc = st.number_input(
        "WBC", 0.1, 200.0, 6.5, 0.1, label_visibility="collapsed", key="wbc"
    )
    st.caption(_RANGE_HINT["WBC"])

    st.markdown("<small>MONOCYTES (G/L)</small>", unsafe_allow_html=True)
    monocytes = st.number_input(
        "MONOCYTES", 0.0, 50.0, 1.0, 0.1, label_visibility="collapsed", key="mon"
    )
    st.caption(_RANGE_HINT["MONOCYTES"])

    st.markdown("<small>PLT (G/L)</small>", unsafe_allow_html=True)
    plt_count = st.number_input(
        "PLT", 2.0, 600.0, 167.0, 1.0, label_visibility="collapsed", key="plt"
    )
    st.caption(_RANGE_HINT["PLT"])

st.sidebar.markdown("---")
st.sidebar.markdown(T("sb_mol"))
mol_choice = st.sidebar.radio(
    T("sb_scenario"),
    options=list(MOLECULAR_PROFILES.keys()),
    format_func=mol_label,
    label_visibility="collapsed",
)
st.sidebar.caption(f"ℹ️ {mol_description(mol_choice)}")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(
    T("sb_predict"), type="primary", use_container_width=True
)

# Keep the prediction visible after other interactions (e.g. clicking Explain)
if predict_btn:
    st.session_state["show_prediction"] = True


# Risk badge

favorable = ["APL t(15;17)", "inv(16)", "t(8;21)", "Normal"]
adverse = ["Monosomy 7", "Complex"]
if cyto_class in favorable:
    risk_badge = T("risk_fav")
    risk_color = "#16a34a"
elif cyto_class in adverse:
    risk_badge = T("risk_adv")
    risk_color = "#dc2626"
else:
    risk_badge = T("risk_int")
    risk_color = "#d97706"

cyto_shown = CYTO_DISPLAY[LANG].get(cyto_display, cyto_display)


# Patient summary + Model info

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.subheader(T("sum_title"))
    c1, c2 = st.columns(2)
    c1.metric(T("sum_cyto_class"), cyto_shown)
    c1.metric(T("sum_eln"), risk_badge)
    c2.metric(T("sum_blasts"), f"{bm_blast:.1f} %")
    c2.metric(T("sum_wbc"), f"{wbc:.1f} G/L")

    rows = []
    for feat, val, unit in [
        ("BM_BLAST", bm_blast, "%"),
        ("WBC", wbc, "G/L"),
        ("ANC", anc, "G/L"),
        ("MONOCYTES", monocytes, "G/L"),
        ("HB", hb, "g/dL"),
        ("PLT", plt_count, "G/L"),
    ]:
        lo, hi = NORMAL_RANGES[feat]
        status = (
            T("st_normal")
            if lo <= val <= hi
            else (T("st_high") if val > hi else T("st_low"))
        )
        rows.append(
            {
                T("col_marker"): FEATURE_LABELS[feat],
                T("col_value"): val,
                T("col_unit"): unit,
                T("col_status"): status,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption(T("sum_mol_scenario", label=mol_label(mol_choice)))

with col2:
    st.subheader(T("models_title"))
    st.markdown(f"""
| | {T("models_clin")} | {T("models_full")} |
|---|---|---|
| **{T("models_algo")}** | Random Survival Forest | Random Survival Forest |
| **{T("models_cindex")}** | {CINDEX_CLINICAL:.3f} | {CINDEX_FULL:.3f} |
""")
    st.caption(T("models_caption"))


# Prediction

if st.session_state.get("show_prediction", False):
    clinical_row = build_clinical_row(
        bm_blast, wbc, anc, monocytes, hb, plt_count, cyto_class
    )
    X_clinical = pd.DataFrame([clinical_row])

    molecular_row = build_molecular_row(mol_choice)
    full_row = {**clinical_row, **molecular_row}
    X_full = pd.DataFrame([full_row])

    try:
        surv_clinical = clinical_model.predict_survival_function(X_clinical)[0]
        surv_full = full_model.predict_survival_function(X_full)[0]
    except Exception as e:
        st.error(T("err_predict", err=e))
        st.stop()

    t_clin, p_clin = surv_clinical.x, surv_clinical(surv_clinical.x)
    t_full, p_full = surv_full.x, surv_full(surv_full.x)

    def summarize(times, probs):
        below_50 = times[probs <= 0.5]
        median = below_50[0] if len(below_50) > 0 else None
        t1 = float(probs[times <= 1.0][-1]) if (times <= 1.0).any() else 1.0
        t3 = float(probs[times <= 3.0][-1]) if (times <= 3.0).any() else 1.0
        return t1, t3, median

    t1_clin, t3_clin, med_clin = summarize(t_clin, p_clin)
    t1_full, t3_full, med_full = summarize(t_full, p_full)

    st.divider()

    show_comparison = mol_choice != "none"

    if show_comparison:
        st.subheader(T("pred_title_cmp"))
    else:
        st.subheader(T("pred_title_clin"))
        st.caption(T("pred_hint_clin"))

    mc1, mc2, mc3 = st.columns(3)
    if show_comparison:
        mc1.metric(
            T("m_1y"),
            f"{t1_full:.1%}",
            delta=T("delta_vs_clin", v=(t1_full - t1_clin) * 100),
        )
        mc2.metric(
            T("m_3y"),
            f"{t3_full:.1%}",
            delta=T("delta_vs_clin", v=(t3_full - t3_clin) * 100),
        )
        mc3.metric(
            T("m_median"), T("m_years", v=med_full) if med_full else T("m_beyond")
        )
    else:
        mc1.metric(T("m_1y"), f"{t1_clin:.1%}")
        mc2.metric(T("m_3y"), f"{t3_clin:.1%}")
        mc3.metric(
            T("m_median"), T("m_years", v=med_clin) if med_clin else T("m_beyond")
        )

    # Survival curve
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if show_comparison:
        ax.step(
            t_clin,
            p_clin,
            where="post",
            color="#94a3b8",
            lw=2,
            ls="--",
            label=T("curve_clin", c=CINDEX_CLINICAL),
        )
        ax.step(
            t_full,
            p_full,
            where="post",
            color="#2563eb",
            lw=2.5,
            label=T("curve_full", c=CINDEX_FULL),
        )
        ax.fill_between(t_full, p_full, alpha=0.08, color="#2563eb", step="post")
        title_suffix = mol_label(mol_choice)
    else:
        ax.step(
            t_clin,
            p_clin,
            where="post",
            color="#2563eb",
            lw=2.5,
            label=T("curve_clin", c=CINDEX_CLINICAL),
        )
        ax.fill_between(t_clin, p_clin, alpha=0.08, color="#2563eb", step="post")
        title_suffix = T("curve_nomol")

    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label=T("curve_thresh"))
    ax.set_xlabel(T("curve_x"), fontsize=11)
    ax.set_ylabel(T("curve_y"), fontsize=11)
    ax.set_title(
        T("curve_title", cyto=cyto_shown, badge=risk_badge, suffix=title_suffix),
        fontsize=11,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(T("curve_cap_cmp") if show_comparison else T("curve_cap_clin"))

    # Clinical interpretation
    st.subheader(T("interp_title"))

    ref_t1 = t1_full if show_comparison else t1_clin
    ref_t3 = t3_full if show_comparison else t3_clin
    ref_median = med_full if show_comparison else med_clin

    if ref_t1 >= 0.80:
        surv_level = T("lvl_very_fav")
    elif ref_t1 >= 0.65:
        surv_level = T("lvl_fav")
    elif ref_t1 >= 0.50:
        surv_level = T("lvl_mod")
    else:
        surv_level = T("lvl_poor")

    median_text = (
        T("interp_median", v=ref_median) if ref_median else T("interp_median_beyond")
    )

    if show_comparison:
        molecular_sentence = T(
            "interp_mol_cmp",
            cyto=cyto_shown,
            badge=risk_badge,
            label=mol_label(mol_choice),
            delta=(t1_full - t1_clin) * 100,
            cf=CINDEX_FULL,
            cc=CINDEX_CLINICAL,
        )
    else:
        molecular_sentence = T(
            "interp_mol_clin",
            cyto=cyto_shown,
            badge=risk_badge,
            cf=CINDEX_FULL,
            cc=CINDEX_CLINICAL,
        )

    main_sentence = T(
        "interp_main", level=surv_level, t1=ref_t1, t3=ref_t3, median=median_text
    )

    st.markdown(
        f"""
<div style="border-left:4px solid {risk_color}; padding:14px 16px; border-radius:0 6px 6px 0; margin-bottom:12px;">
{main_sentence}<br><br>
{molecular_sentence}
</div>
""",
        unsafe_allow_html=True,
    )

    # Key clinical factors
    st.subheader(T("mk_title"))
    st.caption(T("mk_caption"))

    marker_data = [
        ("BM_BLAST", bm_blast, "%", "risk", T("mk_blast")),
        ("WBC", wbc, "G/L", "risk", T("mk_wbc")),
        ("MONOCYTES", monocytes, "G/L", "risk", T("mk_mono")),
        ("ANC", anc, "G/L", "risk", T("mk_anc")),
        ("HB", hb, "g/dL", "protective", T("mk_hb")),
        ("PLT", plt_count, "G/L", "protective", T("mk_plt")),
    ]

    # Cytogenetics is the dominant prognostic factor (ELN 2022): shown first,
    # otherwise the table omits the model's most decisive variable.
    marker_rows = [
        {
            T("col_marker"): T("mk_cyto_row"),
            T("mk_col_effect"): T("mk_cyto_effect"),
            T("mk_col_patient"): f"{risk_badge} — {cyto_shown}",
        }
    ]
    for feat, val, unit, direction, meaning in marker_data:
        lo, hi = NORMAL_RANGES[feat]
        effect_icon = "🔴" if direction == "risk" else "🟢"
        if val < lo:
            patient_status = T("mk_low", v=val, u=unit, lo=lo, hi=hi)
        elif val > hi:
            patient_status = T("mk_high", v=val, u=unit, lo=lo, hi=hi)
        else:
            patient_status = T("mk_normal", v=val, u=unit)

        marker_rows.append(
            {
                T("col_marker"): FEATURE_LABELS[feat],
                T("mk_col_effect"): f"{effect_icon}  {meaning}",
                T("mk_col_patient"): patient_status,
            }
        )

    st.dataframe(pd.DataFrame(marker_rows), hide_index=True, use_container_width=True)
    st.caption(T("mk_legend"))

    if show_comparison and MOLECULAR_PROFILES[mol_choice]["genes"]:
        st.caption(
            T(
                "mk_mol",
                genes=", ".join(MOLECULAR_PROFILES[mol_choice]["genes"]),
                kind=(
                    T("kind_adverse")
                    if mol_choice == "adverse"
                    else T("kind_favourable")
                ),
            )
        )

else:
    st.info(T("idle"))


# Explainability: why does the model predict this? (SHAP)

st.divider()
st.subheader(T("xai_title"))

# Explain the SAME model as shown above: clinical-only when no molecular data.
_use_full = mol_choice != "none"
_model_kind = "full" if _use_full else "clinical"
_model_name = T("xai_model_full") if _use_full else T("xai_model_clin")

with st.expander(T("xai_help_title")):
    st.markdown(T("xai_help_body"))

st.caption(T("xai_model_caption", model=_model_name))

ex_c1, ex_c2, ex_c3 = st.columns([2, 2, 1])
exp_horizon = ex_c1.slider(T("xai_horizon"), 0.5, 10.0, 2.0, 0.5, key="shap_h")
exp_grouped = ex_c2.toggle(T("xai_grouped"), value=True, key="shap_g")

with st.expander(T("xai_settings")):
    exp_bg = st.slider(T("xai_bg"), 10, 60, 25, 5, key="shap_bg")
    exp_ns = st.slider(T("xai_ns"), 50, 300, 120, 10, key="shap_ns")
    st.caption(T("xai_settings_cap"))

if st.button(T("xai_button"), use_container_width=True):
    clinical_row = build_clinical_row(
        bm_blast, wbc, anc, monocytes, hb, plt_count, cyto_class
    )
    full_row = {**clinical_row, **build_molecular_row(mol_choice)}
    row_items = tuple(sorted(full_row.items()))

    with st.spinner(T("xai_spinner")):
        try:
            r = compute_patient_shap(
                row_items, exp_horizon, exp_bg, exp_ns, model_kind=_model_kind
            )
        except FileNotFoundError:
            st.error(T("xai_nobg"))
            st.stop()

    sv, cols = np.array(r["shap"]), r["cols"]
    base, s_patient, t_star = r["base"], r["s_patient"], r["t_star"]

    m1, m2 = st.columns(2)
    m1.metric(
        T("xai_m_pred", t=t_star),
        f"{s_patient:.1%}",
        delta=T("xai_delta", v=(s_patient - base) * 100),
    )
    m2.metric(T("xai_m_base"), f"{base:.1%}")

    if exp_grouped:
        agg = {}
        for v, n in zip(sv, cols):
            fam = feature_family(n)
            agg[fam] = agg.get(fam, 0.0) + v
        s = pd.Series(agg).sort_values(key=np.abs)
        labels, vals = list(s.index), s.values
    else:
        top = np.argsort(np.abs(sv))[::-1][:20]
        order = top[np.argsort(sv[top])]

        def _labelled(col):
            """Label + this patient's value (avoids reading an absent mutation
            in green as if it were present)."""
            lab = pretty_label(col)
            val = full_row.get(col, None)
            if val is None:
                return lab
            if (
                col.startswith("GENE_")
                or col.startswith("EFFECT_")
                or col == "HAS_MOL_DATA"
            ):
                yn = T("lab_yes") if float(val) >= 0.5 else T("lab_no")
                return f"{lab}: {yn}"
            if isinstance(val, str):
                return f"{lab}: {val}"
            return f"{lab} = {val:g}"

        labels, vals = [_labelled(cols[i]) for i in order], sv[order]

    fig_s, ax_s = plt.subplots(figsize=(9, max(3.2, 0.42 * len(labels) + 0.6)))
    ax_s.barh(labels, vals, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in vals])
    ax_s.axvline(0, color="black", lw=0.8)
    ax_s.set_xlabel(T("xai_xlabel", t=t_star))
    ax_s.spines[["top", "right"]].set_visible(False)

    # Real colour patches: emoji are not rendered by matplotlib's default font
    ax_s.legend(
        handles=[
            Patch(facecolor="#2a9d8f", label=T("xai_up")),
            Patch(facecolor="#e76f51", label=T("xai_down")),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()
    st.pyplot(fig_s)
    plt.close(fig_s)

    recon = base + sv.sum()
    if not exp_grouped:
        _shown = [cols[i] for i in order]
        _absent_gene = next(
            (
                c
                for c in reversed(_shown)
                if c.startswith("GENE_") and float(full_row.get(c, 0)) < 0.5
            ),
            None,
        )
        if _absent_gene:
            _example = T("xai_read_gene", g=_absent_gene[5:])
        else:
            _top_col = _shown[int(np.argmax(np.abs(vals)))]
            _example = T("xai_read_generic", lab=_labelled(_top_col))
        st.info(T("xai_read_head") + _example)

    st.caption(T("xai_sanity", recon=recon, s=s_patient))
