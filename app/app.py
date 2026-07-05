"""
app.py — Leukemia Survival Predictor
Usage: streamlit run app/app.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from src.config import CLINICAL_RSF, FULL_RSF


from src.preprocessing import log1p_cols, log1p_full

st.set_page_config(
    page_title="Leukemia Survival Predictor",
    page_icon="🩺",
    layout="wide",
)


# Constants — clinical

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

CYTO_DESCRIPTIONS = {
    "APL t(15;17)": "Translocation t(15;17) — Acute Promyelocytic Leukemia (APL). Best prognosis, responds to targeted therapy (ATRA).",
    "inv(16)": "Inversion of chromosome 16. Favourable prognosis, good response to chemotherapy.",
    "t(8;21)": "Translocation t(8;21). Favourable prognosis, good response to chemotherapy.",
    "Normal": "Normal Caryotype — no chromosomal abnormality detected. Intermediate prognosis.",
    "Trisomy 8": "Extra copy of chromosome 8. Intermediate prognosis.",
    "5q deletion": "Deletion of chromosome 5q. Intermediate-to-adverse prognosis.",
    "Monosomy 7": "Loss of chromosome 7. Adverse prognosis, poor response to standard treatment.",
    "Complex (≥3 anomalies)": "3 or more chromosomal abnormalities detected. Most adverse prognosis, often resistant to standard chemotherapy.",
}

NORMAL_RANGES = {
    "BM_BLAST": (0, 5, "< 5% normal. ≥ 20% = AML diagnosis."),
    "WBC": (4, 10, "Normal: 4–10 G/L"),
    "ANC": (1.8, 7, "Normal: 1.8–7 G/L"),
    "MONOCYTES": (0.2, 1, "Normal: 0.2–1 G/L"),
    "HB": (12, 16, "Normal: 12–16 g/dL"),
    "PLT": (150, 400, "Normal: 150–400 G/L"),
}

FEATURE_LABELS = {
    "BM_BLAST": "Bone marrow blasts",
    "WBC": "White blood cells",
    "ANC": "Neutrophils (ANC)",
    "MONOCYTES": "Monocytes",
    "HB": "Haemoglobin",
    "PLT": "Platelets",
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
# Built to demonstrate how molecular markers shift the survival prediction.
MOLECULAR_PROFILES = {
    "none": {
        "label": "No molecular data available",
        "description": "Patient without molecular testing performed at diagnosis. "
        "The model relies on clinical and cytogenetic data only.",
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
        "label": "Favourable marker — CEBPA mutation",
        "description": "Single CEBPA mutation — a marker associated with favourable "
        "prognosis in AML under ELN 2022 guidelines.",
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
        "label": "Adverse marker — TP53 mutation",
        "description": "TP53 mutation — one of the most well-documented adverse "
        "prognostic markers in AML.",
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

# C-index figures: both from the "Test platform" (held-out) evaluation,
# same methodology for both models. See project README / notebook summary table.
CINDEX_CLINICAL = 0.65
CINDEX_FULL = 0.752


# Model loading


@st.cache_resource(show_spinner="Loading clinical model...")
def load_clinical_model():
    return joblib.load(CLINICAL_RSF)


@st.cache_resource(show_spinner="Loading combined clinical + molecular model...")
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

st.title("🩺 Leukemia Survival Predictor")

st.markdown(
    """
<div style="background-color:#1e3a5f; padding:16px; border-radius:8px; margin-bottom:16px; color:#eef2fa !important;">
<b style="color:#eef2fa;">What is this tool?</b><br>
<span style="color:#eef2fa;">This application estimates the survival probability of a patient diagnosed with
<b style="color:#eef2fa;">Acute Myeloid Leukemia (AML)</b> based on their clinical profile at diagnosis,
and shows how adding molecular data refines the prediction.
It uses <b style="color:#eef2fa;">Random Survival Forest</b> models trained on <b style="color:#eef2fa;">3,323 AML patients</b>
across 10 European centres.</span><br><br>
<b style="color:#eef2fa;">Who is it for?</b> <span style="color:#eef2fa;">Clinicians, researchers, and students in haematology/oncology.</span><br>
<b style="color:#eef2fa;">⚠️ Important:</b> <span style="color:#eef2fa;">For <u>research and educational purposes only</u>.
Must not replace clinical judgement or be used for treatment decisions. Molecular profiles below are
illustrative synthetic examples, not real patient data.</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("📖 How to use — 3 steps", expanded=False):
    st.markdown("""
**Step 1 — Cytogenetic risk group & hematological markers**
Select the Caryotype classification and enter the blood count / bone marrow values.

**Step 2 — Molecular profile**
Choose a molecular scenario: no molecular data, a favourable marker (CEBPA),
or an adverse marker (TP53) — to see how molecular information shifts the prediction.

**Step 3 — Click "Predict survival"**
The app compares the clinical-only prediction against the clinical + molecular
prediction, side by side.
""")

st.divider()

if not (clinical_loaded and full_loaded):
    st.error(
        f"Model not found. Please run the training notebook first.\n\n{load_error}"
    )
    st.stop()


# Sidebar - clinical inputs

st.sidebar.header("🩺 Patient clinical profile")

st.sidebar.markdown("**🧬 Cytogenetic risk group**")
cyto_display = st.sidebar.selectbox(
    "Caryotype (ELN 2022)",
    CYTO_CLASSES,
    index=CYTO_CLASSES.index("Normal"),
    label_visibility="collapsed",
)
cyto_class = CYTO_LABELS_MAP.get(cyto_display, cyto_display)
st.sidebar.caption(f"ℹ️ {CYTO_DESCRIPTIONS[cyto_display]}")

st.sidebar.markdown("---")
st.sidebar.markdown("**🩸 Hematological markers**")

col_a, col_b = st.sidebar.columns(2)

with col_a:
    st.markdown("<small>BM_BLAST (%)</small>", unsafe_allow_html=True)
    bm_blast = st.number_input(
        "BM_BLAST", 0.0, 100.0, 5.0, 0.5, label_visibility="collapsed", key="bm"
    )
    st.caption(NORMAL_RANGES["BM_BLAST"][2])

    st.markdown("<small>ANC (G/L)</small>", unsafe_allow_html=True)
    anc = st.number_input(
        "ANC", 0.0, 120.0, 3.0, 0.1, label_visibility="collapsed", key="anc"
    )
    st.caption(NORMAL_RANGES["ANC"][2])

    st.markdown("<small>HB (g/dL)</small>", unsafe_allow_html=True)
    hb = st.number_input(
        "HB", 4.0, 18.0, 10.0, 0.1, label_visibility="collapsed", key="hb"
    )
    st.caption(NORMAL_RANGES["HB"][2])

with col_b:
    st.markdown("<small>WBC (G/L)</small>", unsafe_allow_html=True)
    wbc = st.number_input(
        "WBC", 0.1, 200.0, 6.5, 0.1, label_visibility="collapsed", key="wbc"
    )
    st.caption(NORMAL_RANGES["WBC"][2])

    st.markdown("<small>MONOCYTES (G/L)</small>", unsafe_allow_html=True)
    monocytes = st.number_input(
        "MONOCYTES", 0.0, 50.0, 1.0, 0.1, label_visibility="collapsed", key="mon"
    )
    st.caption(NORMAL_RANGES["MONOCYTES"][2])

    st.markdown("<small>PLT (G/L)</small>", unsafe_allow_html=True)
    plt_count = st.number_input(
        "PLT", 2.0, 600.0, 167.0, 1.0, label_visibility="collapsed", key="plt"
    )
    st.caption(NORMAL_RANGES["PLT"][2])

st.sidebar.markdown("---")
st.sidebar.markdown("**🧬 Molecular profile**")
mol_choice = st.sidebar.radio(
    "Choose a scenario",
    options=list(MOLECULAR_PROFILES.keys()),
    format_func=lambda k: MOLECULAR_PROFILES[k]["label"],
    label_visibility="collapsed",
)
st.sidebar.caption(f"ℹ️ {MOLECULAR_PROFILES[mol_choice]['description']}")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(
    "🔮 Predict survival", type="primary", use_container_width=True
)


# Risk badge

favorable = ["APL t(15;17)", "inv(16)", "t(8;21)", "Normal"]
adverse = ["Monosomy 7", "Complex"]
if cyto_class in favorable:
    risk_badge = "🟢 Favourable"
    risk_color = "#16a34a"
elif cyto_class in adverse:
    risk_badge = "🔴 Adverse"
    risk_color = "#dc2626"
else:
    risk_badge = "🟡 Intermediate"
    risk_color = "#d97706"


# Patient summary + Model info

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.subheader("Patient summary")
    c1, c2 = st.columns(2)
    c1.metric("Cytogenetic class", cyto_class)
    c1.metric("ELN risk group", risk_badge)
    c2.metric("BM blasts", f"{bm_blast:.1f} %")
    c2.metric("WBC", f"{wbc:.1f} G/L")

    rows = []
    for feat, val, unit in [
        ("BM_BLAST", bm_blast, "%"),
        ("WBC", wbc, "G/L"),
        ("ANC", anc, "G/L"),
        ("MONOCYTES", monocytes, "G/L"),
        ("HB", hb, "g/dL"),
        ("PLT", plt_count, "G/L"),
    ]:
        lo, hi, _ = NORMAL_RANGES[feat]
        status = (
            "✅ Normal" if lo <= val <= hi else ("⚠️ High" if val > hi else "⚠️ Low")
        )
        rows.append(
            {
                "Marker": FEATURE_LABELS[feat],
                "Value": val,
                "Unit": unit,
                "Status": status,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption(f"🧬 Molecular scenario: **{MOLECULAR_PROFILES[mol_choice]['label']}**")

with col2:
    st.subheader("About the models")
    st.markdown(f"""
| | Clinical only | Clinical + molecular |
|---|---|---|
| **Algorithm** | Random Survival Forest | Random Survival Forest |
| **C-index (test set)** | {CINDEX_CLINICAL:.3f} | {CINDEX_FULL:.3f} |
""")
    st.caption(
        "Discrimination score (C-index) on the held-out test platform: 0.5 = random ranking, "
        "1.0 = perfect ranking. Training dataset: 3,323 AML patients, 10 European centres, "
        "ELN 2022 cytogenetic classification."
    )


# Prediction

if predict_btn:
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
        st.error(f"Prediction error: {e}")
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
        st.subheader("Predicted survival — clinical only vs clinical + molecular")
    else:
        st.subheader("Predicted survival — clinical only")
        st.caption(
            "👈 Select a molecular scenario (favourable or adverse marker) in the sidebar "
            "to see how molecular data would refine this prediction."
        )

    if show_comparison:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "1-year survival",
            f"{t1_full:.1%}",
            delta=f"{(t1_full - t1_clin) * 100:+.1f} pts vs clinical only",
        )
        mc2.metric(
            "3-year survival",
            f"{t3_full:.1%}",
            delta=f"{(t3_full - t3_clin) * 100:+.1f} pts vs clinical only",
        )
        mc3.metric(
            "Estimated median survival",
            f"{med_full:.1f} years" if med_full else "> follow-up period",
        )
    else:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("1-year survival", f"{t1_clin:.1%}")
        mc2.metric("3-year survival", f"{t3_clin:.1%}")
        mc3.metric(
            "Estimated median survival",
            f"{med_clin:.1f} years" if med_clin else "> follow-up period",
        )

    # Survival curve — comparison or clinical-only
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if show_comparison:
        ax.step(
            t_clin,
            p_clin,
            where="post",
            color="#94a3b8",
            lw=2,
            ls="--",
            label=f"Clinical only (C-index {CINDEX_CLINICAL:.3f})",
        )
        ax.step(
            t_full,
            p_full,
            where="post",
            color="#2563eb",
            lw=2.5,
            label=f"Clinical + molecular (C-index {CINDEX_FULL:.3f})",
        )
        ax.fill_between(t_full, p_full, alpha=0.08, color="#2563eb", step="post")
        title_suffix = MOLECULAR_PROFILES[mol_choice]["label"]
    else:
        ax.step(
            t_clin,
            p_clin,
            where="post",
            color="#2563eb",
            lw=2.5,
            label=f"Clinical only (C-index {CINDEX_CLINICAL:.3f})",
        )
        ax.fill_between(t_clin, p_clin, alpha=0.08, color="#2563eb", step="post")
        title_suffix = "no molecular data"

    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="50% threshold")
    ax.set_xlabel("Time since diagnosis (years)", fontsize=11)
    ax.set_ylabel("Probability of being alive", fontsize=11)
    ax.set_title(
        f"Survival curve — {cyto_class} · {risk_badge} · {title_suffix}", fontsize=11
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    if show_comparison:
        st.caption(
            "The dashed grey curve uses clinical and cytogenetic data only. The solid blue curve "
            "adds the molecular profile selected in the sidebar — the gap between the two illustrates "
            "the added value of molecular testing at diagnosis."
        )
    else:
        st.caption(
            "This curve uses clinical and cytogenetic data only — no molecular profile has been selected."
        )

    # Clinical interpretation
    st.subheader("🩺 Clinical interpretation")

    ref_t1 = t1_full if show_comparison else t1_clin
    ref_t3 = t3_full if show_comparison else t3_clin
    ref_median = med_full if show_comparison else med_clin

    surv_level = (
        "very favourable"
        if ref_t1 >= 0.80
        else (
            "favourable" if ref_t1 >= 0.65 else "moderate" if ref_t1 >= 0.50 else "poor"
        )
    )
    median_text = (
        f"an estimated median survival of **{ref_median:.1f} years**"
        if ref_median
        else "a median survival that exceeds the observation period (> 10 years)"
    )

    if show_comparison:
        molecular_sentence = (
            f"The cytogenetic class <b>{cyto_class}</b> ({risk_badge}) combined with the molecular scenario "
            f'"<b>{MOLECULAR_PROFILES[mol_choice]["label"]}</b>" shifts the prediction by '
            f"<b>{(t1_full - t1_clin) * 100:+.1f} points</b> at 1 year compared to clinical data alone — "
            f"illustrating why molecular profiling improves prognostic accuracy "
            f"(C-index {CINDEX_FULL:.3f} vs {CINDEX_CLINICAL:.3f})."
        )
    else:
        molecular_sentence = (
            f"The cytogenetic class <b>{cyto_class}</b> ({risk_badge}) is the most influential "
            f"factor in this prediction, consistent with ELN 2022 clinical guidelines. "
            f"Select a molecular scenario in the sidebar to see how molecular testing would "
            f"refine this estimate (C-index {CINDEX_FULL:.3f} vs {CINDEX_CLINICAL:.3f} clinical-only)."
        )

    st.markdown(
        f"""
<div style="border-left:4px solid {risk_color}; padding:14px 16px; border-radius:0 6px 6px 0; margin-bottom:12px;">
This patient has a <b>{surv_level} short-term prognosis</b>, with a
<b>{ref_t1:.0%} probability of survival at 1 year</b> and <b>{ref_t3:.0%} at 3 years</b>,
and {median_text}.<br><br>
{molecular_sentence}
</div>
""",
        unsafe_allow_html=True,
    )

    # Key clinical factors
    st.subheader("🔍 Which clinical markers matter most?")
    st.caption(
        "This table shows how each clinical marker influences survival in AML, "
        "and how this patient compares to normal reference values."
    )

    marker_data = [
        ("BM_BLAST", bm_blast, "%", "risk", "Higher blast % = worse prognosis"),
        ("WBC", wbc, "G/L", "risk", "Higher WBC = worse prognosis"),
        ("MONOCYTES", monocytes, "G/L", "risk", "Higher monocytes = worse prognosis"),
        ("ANC", anc, "G/L", "risk", "Higher neutrophils = worse prognosis"),
        ("HB", hb, "g/dL", "protective", "Higher haemoglobin = better prognosis"),
        ("PLT", plt_count, "G/L", "protective", "Higher platelets = better prognosis"),
    ]

    marker_rows = []
    for feat, val, unit, direction, meaning in marker_data:
        lo, hi, _ = NORMAL_RANGES[feat]
        effect_icon = "🔴" if direction == "risk" else "🟢"
        if val < lo:
            patient_status = f"⚠️ Low — {val} {unit} (normal: {lo}–{hi})"
        elif val > hi:
            patient_status = f"⚠️ High — {val} {unit} (normal: {lo}–{hi})"
        else:
            patient_status = f"✅ Normal — {val} {unit}"

        marker_rows.append(
            {
                "Marker": FEATURE_LABELS[feat],
                "Effect on survival": f"{effect_icon}  {meaning}",
                "This patient": patient_status,
            }
        )

    st.dataframe(pd.DataFrame(marker_rows), hide_index=True, use_container_width=True)

    st.caption(
        "🔴 Risk factors: higher values are associated with worse prognosis. "
        "🟢 Protective factors: higher values are associated with better prognosis. "
        "⚠️ Values outside the normal reference range for this patient. "
        "Based on univariate Cox analysis of clinical markers."
    )

    if show_comparison and MOLECULAR_PROFILES[mol_choice]["genes"]:
        st.caption(
            f"🧬 Molecular marker in this scenario: "
            f"**{', '.join(MOLECULAR_PROFILES[mol_choice]['genes'])}** "
            f"({'adverse' if mol_choice == 'adverse' else 'favourable'} prognostic association)."
        )

else:
    st.info(
        "👈 Enter the patient's clinical profile and choose a molecular scenario in the sidebar, "
        "then click **Predict survival** to compare both models."
    )
