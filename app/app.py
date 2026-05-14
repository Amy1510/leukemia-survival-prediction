"""
app.py — Leukemia Survival Predictor
Usage: streamlit run app/app.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import joblib

from src.features import CYTO_CLASS_ORDER
from src.config import CLINICAL_RSF
from src.preprocessing import log1p_cols, log1p_full

st.set_page_config(
    page_title="Leukemia Survival Predictor",
    page_icon="🩺",
    layout="wide",
)


# Constants

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

# Clinical feature importance from univariate Cox analysis (T2)
# HR values converted to risk contribution (HR-1 for risk factors, 1-HR for protective)
CLINICAL_IMPORTANCE = {
    "Bone marrow blasts": 0.352,  # HR=1.352
    "White blood cells": 0.152,  # HR=1.152
    "Monocytes": 0.116,  # HR=1.116
    "Neutrophils (ANC)": 0.114,  # HR=1.114
    "Platelets": 0.275,  # HR=0.725 → protective
    "Haemoglobin": 0.296,  # HR=0.704 → protective
}

IMPORTANCE_DIRECTION = {
    "Bone marrow blasts": "risk",
    "White blood cells": "risk",
    "Monocytes": "risk",
    "Neutrophils (ANC)": "risk",
    "Platelets": "protective",
    "Haemoglobin": "protective",
}


# Load model


@st.cache_resource
def load_model():
    return joblib.load(CLINICAL_RSF)


try:
    model = load_model()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    load_error = str(e)


# Welcome message

st.title("🩺 Leukemia Survival Predictor")

st.markdown(
    """
<div style="background-color:#1e3a5f; padding:16px; border-radius:8px; margin-bottom:16px;">
<b>What is this tool?</b><br>
This application estimates the survival probability of a patient diagnosed with
<b>Acute Myeloid Leukemia (AML)</b> based on their clinical profile at diagnosis.
It uses a <b>Random Survival Forest</b> trained on <b>3,323 AML patients</b>
across 10 European centres.<br><br>
<b>Who is it for?</b> Clinicians, researchers, and students in haematology/oncology.<br>
<b>⚠️ Important:</b> For <u>research and educational purposes only</u>.
Must not replace clinical judgement or be used for treatment decisions.
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("📖 How to use — 3 steps", expanded=False):
    st.markdown("""
**Step 1 — Cytogenetic risk group**
Select the Caryotype classification from the cytogenetic analysis report (available at AML diagnosis).

**Step 2 — Hematological markers**
Enter the values from the patient's blood count (NFS) and bone marrow aspirate at diagnosis.

**Step 3 — Click "Predict survival"**
The model returns the predicted survival curve, key probabilities at 1 and 3 years,
estimated median survival, and a clinical interpretation.
""")

st.divider()

if not model_loaded:
    st.error(
        f"Model not found. Please run the training notebook first.\n\n{load_error}"
    )
    st.stop()


# Sidebar — compact layout

st.sidebar.header("🩺 Patient clinical profile")

# Cytogenetics
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

# Compact number inputs — 2 columns
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

with col2:
    st.subheader("About the model")
    st.markdown("""
| | |
|---|---|
| **Algorithm** | Random Survival Forest |
| **Training dataset** | 3,323 AML patients, 10 European centres |
| **Cytogenetic classification** | ELN 2022 standards |
""")
    st.caption(
        "Discrimination score: 0.716 on clinical features alone "
        "(0.5 = random ranking, 1.0 = perfect ranking)."
    )


# Prediction

if predict_btn:
    X_df = pd.DataFrame(
        [
            [
                bm_blast,
                wbc,
                anc,
                monocytes,
                hb,
                plt_count,
                cyto_class,
                "KI",
            ]
        ],
        columns=[
            "BM_BLAST",
            "WBC",
            "ANC",
            "MONOCYTES",
            "HB",
            "PLT",
            "CYTO_CLASS",
            "CENTER",
        ],
    )

    try:
        surv_fn = model.predict_survival_function(X_df)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    times, probs = surv_fn.x, surv_fn(surv_fn.x)
    below_50 = times[probs <= 0.5]
    median_surv = below_50[0] if len(below_50) > 0 else None
    t1 = float(probs[times <= 1.0][-1]) if (times <= 1.0).any() else 1.0
    t3 = float(probs[times <= 3.0][-1]) if (times <= 3.0).any() else 1.0

    st.divider()
    st.subheader("Predicted survival")

    m1, m2, m3 = st.columns(3)
    m1.metric("1-year survival probability", f"{t1:.1%}")
    m2.metric("3-year survival probability", f"{t3:.1%}")
    m3.metric(
        "Estimated median survival",
        f"{median_surv:.1f} years" if median_surv else "> follow-up period",
    )

    # Survival curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(
        times, probs, where="post", color="#2563eb", lw=2.5, label="Predicted survival"
    )
    ax.fill_between(times, probs, alpha=0.1, color="#2563eb", step="post")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="50% threshold")
    if median_surv:
        ax.axvline(
            median_surv,
            color="#dc2626",
            ls=":",
            lw=1.5,
            label=f"Median ≈ {median_surv:.1f} yr",
        )
    ax.set_xlabel("Time since diagnosis (years)", fontsize=11)
    ax.set_ylabel("Probability of being alive", fontsize=11)
    ax.set_title(f"Survival curve — {cyto_class} · {risk_badge}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Clinical interpretation
    st.subheader("🩺 Clinical interpretation")

    surv_level = (
        "very favourable"
        if t1 >= 0.80
        else "favourable" if t1 >= 0.65 else "moderate" if t1 >= 0.50 else "poor"
    )
    median_text = (
        f"an estimated median survival of **{median_surv:.1f} years**"
        if median_surv
        else "a median survival that exceeds the observation period (> 10 years)"
    )

    st.markdown(
        f"""
<div style="border-left:4px solid {risk_color}; padding:14px 16px; border-radius:0 6px 6px 0; margin-bottom:12px;">
This patient has a <b>{surv_level} short-term prognosis</b>, with a
<b>{t1:.0%} probability of survival at 1 year</b> and <b>{t3:.0%} at 3 years</b>,
and {median_text}.<br><br>
The cytogenetic class <b>{cyto_class}</b> ({risk_badge}) is the most influential
factor in this prediction, consistent with ELN 2022 clinical guidelines.
</div>
""",
        unsafe_allow_html=True,
    )

    # Key factors — simple readable table
    st.subheader("🔍 Which markers matter most?")
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

    rows = []
    for feat, val, unit, direction, meaning in marker_data:
        lo, hi, _ = NORMAL_RANGES[feat]
        effect_icon = "🔴" if direction == "risk" else "🟢"
        if val < lo:
            patient_status = f"⚠️ Low — {val} {unit} (normal: {lo}–{hi})"
        elif val > hi:
            patient_status = f"⚠️ High — {val} {unit} (normal: {lo}–{hi})"
        else:
            patient_status = f"✅ Normal — {val} {unit}"

        rows.append(
            {
                "Marker": FEATURE_LABELS[feat],
                "Effect on survival": f"{effect_icon}  {meaning}",
                "This patient": patient_status,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "🔴 Risk factors: higher values are associated with worse prognosis. "
        "🟢 Protective factors: higher values are associated with better prognosis. "
        "⚠️ Values outside the normal reference range for this patient."
    )

else:
    st.info(
        "👈 Enter the patient's clinical profile in the sidebar "
        "and click **Predict survival** to see the results."
    )
