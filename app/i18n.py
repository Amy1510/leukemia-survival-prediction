"""
i18n.py, English / French translations for the Leukemia Survival Predictor.

Usage in app.py:
    from i18n import t, CYTO_DESC, RANGE_HINTS, FEATURE_LABELS_I18N, MOL_PROFILES_I18N
    lang = "fr"
    st.title(t("app_title", lang))

Placeholders use str.format syntax:  t("delta_vs_clin", lang, pts=3.2)
"""

LANGUAGES = {"en": "🇬🇧 English", "fr": "🇫🇷 Français"}


def t(key, lang="en", **kwargs):
    """Return the translated string for `key`, falling back to English."""
    s = TR.get(lang, TR["en"]).get(key) or TR["en"].get(key, key)
    return s.format(**kwargs) if kwargs else s


# Cytogenetic descriptions (keyed by the display label used in the selectbox)
CYTO_DESC = {
    "en": {
        "APL t(15;17)": "Translocation t(15;17) — Acute Promyelocytic Leukemia (APL). Best prognosis, responds to targeted therapy (ATRA).",
        "inv(16)": "Inversion of chromosome 16. Favourable prognosis, good response to chemotherapy.",
        "t(8;21)": "Translocation t(8;21). Favourable prognosis, good response to chemotherapy.",
        "Normal": "Normal Caryotype — no chromosomal abnormality detected. Intermediate prognosis.",
        "Trisomy 8": "Extra copy of chromosome 8. Intermediate prognosis.",
        "5q deletion": "Deletion of chromosome 5q. Intermediate-to-adverse prognosis.",
        "Monosomy 7": "Loss of chromosome 7. Adverse prognosis, poor response to standard treatment.",
        "Complex (≥3 anomalies)": "3 or more chromosomal abnormalities detected. Most adverse prognosis, often resistant to standard chemotherapy.",
    },
    "fr": {
        "APL t(15;17)": "Translocation t(15;17) — leucémie aiguë promyélocytaire (LAP). Meilleur pronostic, répond au traitement ciblé (ATRA).",
        "inv(16)": "Inversion du chromosome 16. Pronostic favorable, bonne réponse à la chimiothérapie.",
        "t(8;21)": "Translocation t(8;21). Pronostic favorable, bonne réponse à la chimiothérapie.",
        "Normal": "Caryotype normal — aucune anomalie chromosomique détectée. Pronostic intermédiaire.",
        "Trisomy 8": "Copie supplémentaire du chromosome 8. Pronostic intermédiaire.",
        "5q deletion": "Délétion du chromosome 5q. Pronostic intermédiaire à défavorable.",
        "Monosomy 7": "Perte du chromosome 7. Pronostic défavorable, faible réponse au traitement standard.",
        "Complex (≥3 anomalies)": "3 anomalies chromosomiques ou plus. Pronostic le plus défavorable, souvent résistant à la chimiothérapie standard.",
    },
}

# Cytogenetic display names (only "Complex" needs translating; the rest are
# international nomenclature and stay identical)
CYTO_DISPLAY = {
    "en": {
        "Complex (≥3 anomalies)": "Complex (≥3 anomalies)",
        "Normal": "Normal",
        "Trisomy 8": "Trisomy 8",
        "5q deletion": "5q deletion",
        "Monosomy 7": "Monosomy 7",
    },
    "fr": {
        "Complex (≥3 anomalies)": "Complexe (≥3 anomalies)",
        "Normal": "Normal",
        "Trisomy 8": "Trisomie 8",
        "5q deletion": "Délétion 5q",
        "Monosomy 7": "Monosomie 7",
    },
}

# Reference-range hints shown under each numeric input
RANGE_HINTS = {
    "en": {
        "BM_BLAST": "< 5% normal. ≥ 20% = AML diagnosis.",
        "WBC": "Normal: 4–10 G/L",
        "ANC": "Normal: 1.8–7 G/L",
        "MONOCYTES": "Normal: 0.2–1 G/L",
        "HB": "Normal: 12–16 g/dL",
        "PLT": "Normal: 150–400 G/L",
    },
    "fr": {
        "BM_BLAST": "< 5 % normal. ≥ 20 % = diagnostic de LAM.",
        "WBC": "Normale : 4–10 G/L",
        "ANC": "Normale : 1,8–7 G/L",
        "MONOCYTES": "Normale : 0,2–1 G/L",
        "HB": "Normale : 12–16 g/dL",
        "PLT": "Normale : 150–400 G/L",
    },
}

# Clinical marker labels
FEATURE_LABELS_I18N = {
    "en": {
        "BM_BLAST": "Bone marrow blasts",
        "WBC": "White blood cells",
        "ANC": "Neutrophils (ANC)",
        "MONOCYTES": "Monocytes",
        "HB": "Haemoglobin",
        "PLT": "Platelets",
    },
    "fr": {
        "BM_BLAST": "Blastes médullaires",
        "WBC": "Globules blancs",
        "ANC": "Neutrophiles (PNN)",
        "MONOCYTES": "Monocytes",
        "HB": "Hémoglobine",
        "PLT": "Plaquettes",
    },
}

# Molecular scenarios (label + description)
MOL_PROFILES_I18N = {
    "en": {
        "none": {
            "label": "No molecular data available",
            "description": "Patient without molecular testing performed at diagnosis. "
            "The model relies on clinical and cytogenetic data only.",
        },
        "favorable": {
            "label": "Favourable marker — CEBPA mutation",
            "description": "Single CEBPA mutation — a marker associated with favourable "
            "prognosis in AML under ELN 2022 guidelines.",
        },
        "adverse": {
            "label": "Adverse marker — TP53 mutation",
            "description": "TP53 mutation — one of the most well-documented adverse "
            "prognostic markers in AML.",
        },
    },
    "fr": {
        "none": {
            "label": "Aucune donnée moléculaire",
            "description": "Patient n'ayant pas bénéficié d'un test moléculaire au diagnostic. "
            "Le modèle s'appuie uniquement sur les données cliniques et cytogénétiques.",
        },
        "favorable": {
            "label": "Marqueur favorable — mutation CEBPA",
            "description": "Mutation isolée de CEBPA — marqueur associé à un pronostic "
            "favorable dans la LAM selon les recommandations ELN 2022.",
        },
        "adverse": {
            "label": "Marqueur défavorable — mutation TP53",
            "description": "Mutation de TP53 — l'un des marqueurs pronostiques défavorables "
            "les mieux documentés dans la LAM.",
        },
    },
}

# Feature families (explainability, grouped view)
FAMILIES = {
    "en": {
        "cyto": "Cytogenetics (ELN)",
        "center": "Center",
        "hema": "Hematology",
        "burden": "Mutation burden",
        "genes": "Mutated genes",
        "effects": "Mutation effects",
    },
    "fr": {
        "cyto": "Cytogénétique (ELN)",
        "center": "Centre",
        "hema": "Hématologie",
        "burden": "Charge mutationnelle",
        "genes": "Gènes mutés",
        "effects": "Effets de mutation",
    },
}

# Molecular-burden variable labels (detailed explainability view)
BURDEN_LABELS = {
    "en": {
        "N_MUT": "Number of mutations",
        "N_GENES": "Number of mutated genes",
        "VAF_MEAN": "Mean VAF",
        "VAF_MAX": "Max VAF",
        "DEPTH_MEAN": "Mean sequencing depth",
        "DEPTH_MAX": "Max sequencing depth",
        "HAS_MOL_DATA": "Molecular data available",
    },
    "fr": {
        "N_MUT": "Nombre de mutations",
        "N_GENES": "Nombre de gènes mutés",
        "VAF_MEAN": "VAF moyenne",
        "VAF_MAX": "VAF maximale",
        "DEPTH_MEAN": "Profondeur de séquençage moyenne",
        "DEPTH_MAX": "Profondeur de séquençage maximale",
        "HAS_MOL_DATA": "Données moléculaires disponibles",
    },
}

# Main string table
TR = {
    "en": {
        "lang_label": "Language",
        "app_title": "🩺 Leukemia Survival Predictor",
        "intro_what": "What is this tool?",
        "intro_body": "This application estimates the survival probability of a patient diagnosed with "
        "<b>Acute Myeloid Leukemia (AML)</b> based on their clinical profile at diagnosis, "
        "and shows how adding molecular data refines the prediction. It uses "
        "<b>Random Survival Forest</b> models trained on <b>3,323 AML patients</b> "
        "across 10 European centres.",
        "intro_who": "Who is it for?",
        "intro_who_body": "Clinicians, researchers, and students in haematology/oncology.",
        "intro_warn": "⚠️ Important:",
        "intro_warn_body": "For <u>research and educational purposes only</u>. Must not replace clinical "
        "judgement or be used for treatment decisions. Molecular profiles below are "
        "illustrative synthetic examples, not real patient data.",
        "howto_title": "📖 How to use — 3 steps",
        "howto_body": """
**Step 1 — Cytogenetic risk group & hematological markers**
Select the Caryotype classification and enter the blood count / bone marrow values.

**Step 2 — Molecular profile**
Choose a molecular scenario: no molecular data, a favourable marker (CEBPA),
or an adverse marker (TP53) — to see how molecular information shifts the prediction.

**Step 3 — Click "Predict survival"**
The app compares the clinical-only prediction against the clinical + molecular
prediction, side by side.
""",
        "sb_header": "🩺 Patient clinical profile",
        "sb_cyto": "**🧬 Cytogenetic risk group**",
        "sb_caryotype": "Caryotype (ELN 2022)",
        "sb_hema": "**🩸 Hematological markers**",
        "sb_mol": "**🧬 Molecular profile**",
        "sb_scenario": "Choose a scenario",
        "sb_predict": "🔮 Predict survival",
        "risk_fav": "🟢 Favourable",
        "risk_adv": "🔴 Adverse",
        "risk_int": "🟡 Intermediate",
        "sum_title": "Patient summary",
        "sum_cyto_class": "Cytogenetic class",
        "sum_eln": "ELN risk group",
        "sum_blasts": "BM blasts",
        "sum_wbc": "WBC",
        "col_marker": "Marker",
        "col_value": "Value",
        "col_unit": "Unit",
        "col_status": "Status",
        "st_normal": "✅ Normal",
        "st_high": "⚠️ High",
        "st_low": "⚠️ Low",
        "sum_mol_scenario": "🧬 Molecular scenario: **{label}**",
        "models_title": "About the models",
        "models_clin": "Clinical only",
        "models_full": "Clinical + molecular",
        "models_algo": "Algorithm",
        "models_cindex": "C-index (test set)",
        "models_caption": "Discrimination score (C-index) on the held-out test platform: 0.5 = random "
        "ranking, 1.0 = perfect ranking. Training dataset: 3,323 AML patients, "
        "10 European centres, ELN 2022 cytogenetic classification.",
        "pred_title_cmp": "Predicted survival — clinical only vs clinical + molecular",
        "pred_title_clin": "Predicted survival — clinical only",
        "pred_hint_clin": "👈 Select a molecular scenario (favourable or adverse marker) in the sidebar "
        "to see how molecular data would refine this prediction.",
        "m_1y": "1-year survival",
        "m_3y": "3-year survival",
        "m_median": "Estimated median survival",
        "m_years": "{v:.1f} years",
        "m_beyond": "> follow-up period",
        "delta_vs_clin": "{v:+.1f} pts vs clinical only",
        "curve_clin": "Clinical only (C-index {c:.3f})",
        "curve_full": "Clinical + molecular (C-index {c:.3f})",
        "curve_thresh": "50% threshold",
        "curve_x": "Time since diagnosis (years)",
        "curve_y": "Probability of being alive",
        "curve_title": "Survival curve — {cyto} · {badge} · {suffix}",
        "curve_nomol": "no molecular data",
        "curve_cap_cmp": "The dashed grey curve uses clinical and cytogenetic data only. The solid blue "
        "curve adds the molecular profile selected in the sidebar — the gap between the "
        "two illustrates the added value of molecular testing at diagnosis.",
        "curve_cap_clin": "This curve uses clinical and cytogenetic data only — no molecular profile "
        "has been selected.",
        "interp_title": "🩺 Clinical interpretation",
        "lvl_very_fav": "very favourable",
        "lvl_fav": "favourable",
        "lvl_mod": "moderate",
        "lvl_poor": "poor",
        "interp_median": "an estimated median survival of <b>{v:.1f} years</b>",
        "interp_median_beyond": "a median survival that exceeds the observation period (> 10 years)",
        "interp_main": "This patient has a <b>{level} short-term prognosis</b>, with a "
        "<b>{t1:.0%} probability of survival at 1 year</b> and <b>{t3:.0%} at 3 years</b>, "
        "and {median}.",
        "interp_mol_cmp": "The cytogenetic class <b>{cyto}</b> ({badge}) combined with the molecular "
        'scenario "<b>{label}</b>" shifts the prediction by <b>{delta:+.1f} points</b> '
        "at 1 year compared to clinical data alone — illustrating why molecular "
        "profiling improves prognostic accuracy (C-index {cf:.3f} vs {cc:.3f}).",
        "interp_mol_clin": "The cytogenetic class <b>{cyto}</b> ({badge}) is the most influential factor "
        "in this prediction, consistent with ELN 2022 clinical guidelines. Select a "
        "molecular scenario in the sidebar to see how molecular testing would refine "
        "this estimate (C-index {cf:.3f} vs {cc:.3f} clinical-only).",
        "mk_title": "🔍 Which clinical markers matter most?",
        "mk_caption": "This table shows how each clinical marker influences survival in AML, and how "
        "this patient compares to normal reference values.",
        "mk_col_effect": "Effect on survival",
        "mk_col_patient": "This patient",
        "mk_cyto_row": "Cytogenetic class (ELN)",
        "mk_cyto_effect": "⭐  Strongest prognostic factor — adverse karyotype = worse prognosis",
        "mk_blast": "Higher blast % = worse prognosis",
        "mk_wbc": "Higher WBC = worse prognosis",
        "mk_mono": "Higher monocytes = worse prognosis",
        "mk_anc": "Higher neutrophils = worse prognosis",
        "mk_hb": "Higher haemoglobin = better prognosis",
        "mk_plt": "Higher platelets = better prognosis",
        "mk_low": "⚠️ Low — {v} {u} (normal: {lo}–{hi})",
        "mk_high": "⚠️ High — {v} {u} (normal: {lo}–{hi})",
        "mk_normal": "✅ Normal — {v} {u}",
        "mk_legend": "⭐ Dominant prognostic factor (ELN 2022 cytogenetic classification). "
        "🔴 Risk factors: higher values are associated with worse prognosis. "
        "🟢 Protective factors: higher values are associated with better prognosis. "
        "⚠️ Values outside the normal reference range for this patient. "
        "Based on univariate Cox analysis of clinical markers.",
        "mk_mol": "🧬 Molecular marker in this scenario: **{genes}** ({kind} prognostic association).",
        "kind_adverse": "adverse",
        "kind_favourable": "favourable",
        "xai_title": "🔬 Explainability — why this prediction?",
        "xai_help_title": "❓ How to use the controls below",
        "xai_help_body": """
**Horizon (years)** — the point on the survival curve being explained.
The model predicts a *full curve*, but SHAP needs a single number, so we read survival at
this horizon (e.g. 2 years) and explain that value.
*A feature can matter differently at 1 year and at 5 years, so moving this slider
legitimately changes the explanation — comparing horizons is informative in itself.*
👉 Set it to a clinically meaningful checkpoint: **1–2 years** for short-term prognosis,
**5 years** for long-term outlook.

**Group by family** — display option only, it does not change the computation.
*Off*: one bar per variable, with this patient's value (`Haemoglobin = 10`).
*On*: variables are pooled into families (Hematology, Cytogenetics, Mutated genes,
Mutation burden) and their contributions summed.
👉 Keep it **on** with the molecular model (56 variables would be unreadable), turn it
**off** to see exactly which variable drives the prediction.

**Compute settings** — precision vs. speed and memory.
*Background size*: how many reference patients define the "average patient" baseline.
*nsamples*: how many feature combinations SHAP tests to estimate each contribution.
👉 Defaults (25 / 120) are a good balance. Raise them for **stable, publication-grade
values**; lower them if the app becomes slow or runs out of memory. Note that changing the
background size slightly shifts the baseline, since the reference population changes.

---
Results are **cached** per profile, horizon and settings — re-running the same
configuration is instant.
""",
        "xai_model_caption": "Explaining the **{model}** model (same one shown above), for the profile "
        "currently set in the sidebar.",
        "xai_model_clin": "clinical-only",
        "xai_model_full": "clinical + molecular",
        "xai_horizon": "Horizon (years)",
        "xai_grouped": "Group by family",
        "xai_settings": "⚙️ Compute settings (memory / precision)",
        "xai_bg": "Background size",
        "xai_ns": "nsamples",
        "xai_settings_cap": "Higher = more precise but heavier. If the app runs out of memory, lower "
        "both. The result is cached per (profile, horizon, settings).",
        "xai_button": "🔎 Explain this prediction",
        "xai_spinner": "Computing SHAP contributions…",
        "xai_nobg": "Background file not found. Export it once from the notebook:\n\n"
        "`X_bg.sample(100, random_state=0).to_parquet('models/full/shap_background.parquet')`",
        "xai_m_pred": "Predicted survival at {t:g} y",
        "xai_m_base": "Baseline (average patient)",
        "xai_delta": "{v:+.1f} pts vs average",
        "xai_up": "Pushes survival up",
        "xai_down": "Pushes survival down",
        "xai_xlabel": "Contribution to survival at {t:g} years",
        "xai_read_head": "**How to read this:** each bar shows how *this patient's value* shifts "
        "survival **relative to the average patient**, not whether the feature is "
        "good or bad in itself. ",
        "xai_read_gene": "Example: `{g} mutation: No` in green means the **absence** of a {g} mutation "
        "raises survival compared with a population where some patients carry it — "
        "it does *not* mean the mutation is protective.",
        "xai_read_generic": "Example: `{lab}` is compared with the average value across the training "
        "cohort — a green bar means this patient's value is more favourable than "
        "that average, not that it is normal in absolute terms.",
        "xai_sanity": "Sanity check: baseline + Σ contributions = {recon:.3f} (= predicted survival "
        "{s:.3f} ✓). 🟢 higher haemoglobin / platelets raise survival; 🔴 higher blasts / "
        "adverse cytogenetics lower it — consistent with the univariate Cox analysis.",
        "lab_yes": "Yes",
        "lab_no": "No",
        "lab_mutation": "{g} mutation",
        "lab_effect": "Effect: {e}",
        "lab_cyto": "Cytogenetics (ELN class)",
        "lab_center": "Treatment center",
        "err_nomodel": "Model not found. Please run the training notebook first.\n\n{err}",
        "err_predict": "Prediction error: {err}",
        "idle": "👈 Enter the patient's clinical profile and choose a molecular scenario in the "
        "sidebar, then click **Predict survival** to compare both models.",
        "load_clin": "Loading clinical model...",
        "load_full": "Loading combined clinical + molecular model...",
    },
    "fr": {
        "lang_label": "Langue",
        "app_title": "🩺 Prédicteur de survie — Leucémie aiguë myéloïde",
        "intro_what": "Qu'est-ce que cet outil ?",
        "intro_body": "Cette application estime la probabilité de survie d'un patient atteint de "
        "<b>leucémie aiguë myéloïde (LAM)</b> à partir de son profil clinique au "
        "diagnostic, et montre comment l'ajout de données moléculaires affine la "
        "prédiction. Elle repose sur des modèles <b>Random Survival Forest</b> entraînés "
        "sur <b>3 323 patients</b> répartis dans 10 centres européens.",
        "intro_who": "À qui s'adresse-t-elle ?",
        "intro_who_body": "Cliniciens, chercheurs et étudiants en hématologie/oncologie.",
        "intro_warn": "⚠️ Important :",
        "intro_warn_body": "Usage <u>recherche et pédagogique uniquement</u>. Ne doit pas remplacer le "
        "jugement clinique ni servir à des décisions thérapeutiques. Les profils "
        "moléculaires ci-dessous sont des exemples synthétiques illustratifs, et non "
        "des données patients réelles.",
        "howto_title": "📖 Mode d'emploi — 3 étapes",
        "howto_body": """
**Étape 1 — Groupe de risque cytogénétique & marqueurs hématologiques**
Sélectionnez la classification du caryotype et saisissez les valeurs de l'hémogramme
et du myélogramme.

**Étape 2 — Profil moléculaire**
Choisissez un scénario moléculaire : aucune donnée, un marqueur favorable (CEBPA)
ou un marqueur défavorable (TP53) — pour voir comment l'information moléculaire
modifie la prédiction.

**Étape 3 — Cliquez sur « Prédire la survie »**
L'application compare côte à côte la prédiction clinique seule et la prédiction
clinique + moléculaire.
""",
        "sb_header": "🩺 Profil clinique du patient",
        "sb_cyto": "**🧬 Groupe de risque cytogénétique**",
        "sb_caryotype": "Caryotype (ELN 2022)",
        "sb_hema": "**🩸 Marqueurs hématologiques**",
        "sb_mol": "**🧬 Profil moléculaire**",
        "sb_scenario": "Choisir un scénario",
        "sb_predict": "🔮 Prédire la survie",
        "risk_fav": "🟢 Favorable",
        "risk_adv": "🔴 Défavorable",
        "risk_int": "🟡 Intermédiaire",
        "sum_title": "Résumé du patient",
        "sum_cyto_class": "Classe cytogénétique",
        "sum_eln": "Groupe de risque ELN",
        "sum_blasts": "Blastes médullaires",
        "sum_wbc": "GB",
        "col_marker": "Marqueur",
        "col_value": "Valeur",
        "col_unit": "Unité",
        "col_status": "Statut",
        "st_normal": "✅ Normal",
        "st_high": "⚠️ Élevé",
        "st_low": "⚠️ Bas",
        "sum_mol_scenario": "🧬 Scénario moléculaire : **{label}**",
        "models_title": "À propos des modèles",
        "models_clin": "Clinique seul",
        "models_full": "Clinique + moléculaire",
        "models_algo": "Algorithme",
        "models_cindex": "C-index (jeu de test)",
        "models_caption": "Score de discrimination (C-index) sur la plateforme de test indépendante : "
        "0,5 = classement aléatoire, 1,0 = classement parfait. Jeu d'entraînement : "
        "3 323 patients, 10 centres européens, classification cytogénétique ELN 2022.",
        "pred_title_cmp": "Survie prédite — clinique seul vs clinique + moléculaire",
        "pred_title_clin": "Survie prédite — clinique seul",
        "pred_hint_clin": "👈 Sélectionnez un scénario moléculaire (marqueur favorable ou défavorable) "
        "dans la barre latérale pour voir comment les données moléculaires affinent "
        "cette prédiction.",
        "m_1y": "Survie à 1 an",
        "m_3y": "Survie à 3 ans",
        "m_median": "Survie médiane estimée",
        "m_years": "{v:.1f} ans",
        "m_beyond": "> période de suivi",
        "delta_vs_clin": "{v:+.1f} pts vs clinique seul",
        "curve_clin": "Clinique seul (C-index {c:.3f})",
        "curve_full": "Clinique + moléculaire (C-index {c:.3f})",
        "curve_thresh": "Seuil 50 %",
        "curve_x": "Temps depuis le diagnostic (années)",
        "curve_y": "Probabilité d'être en vie",
        "curve_title": "Courbe de survie — {cyto} · {badge} · {suffix}",
        "curve_nomol": "aucune donnée moléculaire",
        "curve_cap_cmp": "La courbe grise en pointillés utilise uniquement les données cliniques et "
        "cytogénétiques. La courbe bleue pleine ajoute le profil moléculaire "
        "sélectionné — l'écart entre les deux illustre l'apport du test moléculaire "
        "au diagnostic.",
        "curve_cap_clin": "Cette courbe utilise uniquement les données cliniques et cytogénétiques — "
        "aucun profil moléculaire n'a été sélectionné.",
        "interp_title": "🩺 Interprétation clinique",
        "lvl_very_fav": "très favorable",
        "lvl_fav": "favorable",
        "lvl_mod": "modéré",
        "lvl_poor": "défavorable",
        "interp_median": "une survie médiane estimée à <b>{v:.1f} ans</b>",
        "interp_median_beyond": "une survie médiane dépassant la période d'observation (> 10 ans)",
        "interp_main": "Ce patient présente un <b>pronostic {level} à court terme</b>, avec une "
        "<b>probabilité de survie de {t1:.0%} à 1 an</b> et de <b>{t3:.0%} à 3 ans</b>, "
        "et {median}.",
        "interp_mol_cmp": "La classe cytogénétique <b>{cyto}</b> ({badge}) combinée au scénario "
        "moléculaire « <b>{label}</b> » modifie la prédiction de "
        "<b>{delta:+.1f} points</b> à 1 an par rapport aux données cliniques seules — "
        "illustrant pourquoi le profilage moléculaire améliore la précision "
        "pronostique (C-index {cf:.3f} vs {cc:.3f}).",
        "interp_mol_clin": "La classe cytogénétique <b>{cyto}</b> ({badge}) est le facteur le plus "
        "influent de cette prédiction, conformément aux recommandations ELN 2022. "
        "Sélectionnez un scénario moléculaire dans la barre latérale pour voir "
        "comment le test moléculaire affinerait cette estimation "
        "(C-index {cf:.3f} vs {cc:.3f} pour le clinique seul).",
        "mk_title": "🔍 Quels marqueurs cliniques comptent le plus ?",
        "mk_caption": "Ce tableau montre comment chaque marqueur clinique influence la survie dans la "
        "LAM, et comment ce patient se situe par rapport aux valeurs de référence.",
        "mk_col_effect": "Effet sur la survie",
        "mk_col_patient": "Ce patient",
        "mk_cyto_row": "Classe cytogénétique (ELN)",
        "mk_cyto_effect": "⭐  Facteur pronostique dominant — caryotype défavorable = pronostic plus sombre",
        "mk_blast": "Plus de blastes = pronostic plus sombre",
        "mk_wbc": "GB plus élevés = pronostic plus sombre",
        "mk_mono": "Monocytes plus élevés = pronostic plus sombre",
        "mk_anc": "Neutrophiles plus élevés = pronostic plus sombre",
        "mk_hb": "Hémoglobine plus élevée = meilleur pronostic",
        "mk_plt": "Plaquettes plus élevées = meilleur pronostic",
        "mk_low": "⚠️ Bas — {v} {u} (normale : {lo}–{hi})",
        "mk_high": "⚠️ Élevé — {v} {u} (normale : {lo}–{hi})",
        "mk_normal": "✅ Normal — {v} {u}",
        "mk_legend": "⭐ Facteur pronostique dominant (classification cytogénétique ELN 2022). "
        "🔴 Facteurs de risque : des valeurs élevées sont associées à un pronostic plus "
        "sombre. 🟢 Facteurs protecteurs : des valeurs élevées sont associées à un "
        "meilleur pronostic. ⚠️ Valeurs hors de l'intervalle de référence pour ce patient. "
        "D'après l'analyse de Cox univariée des marqueurs cliniques.",
        "mk_mol": "🧬 Marqueur moléculaire de ce scénario : **{genes}** (association pronostique {kind}).",
        "kind_adverse": "défavorable",
        "kind_favourable": "favorable",
        "xai_title": "🔬 Explicabilité — pourquoi cette prédiction ?",
        "xai_help_title": "❓ Comment utiliser les réglages ci-dessous",
        "xai_help_body": """
**Horizon (années)**: le point de la courbe de survie qui est expliqué.
Le modèle prédit une *courbe complète*, mais SHAP a besoin d'un seul nombre : on lit donc
la survie à cet horizon (2 ans par exemple) et c'est cette valeur qui est décomposée.
*Une variable peut peser différemment à 1 an et à 5 ans : déplacer ce curseur change donc
légitimement l'explication, et comparer plusieurs horizons est en soi une information.*
👉 Choisissez un jalon cliniquement parlant : **1–2 ans** pour le pronostic à court terme,
**5 ans** pour le long terme.

**Regrouper par famille**: option d'affichage uniquement, sans effet sur le calcul.
*Désactivé* : une barre par variable, avec la valeur du patient (`Hémoglobine = 10`).
*Activé* : les variables sont regroupées par famille (Hématologie, Cytogénétique,
Gènes mutés, Charge mutationnelle) et leurs contributions additionnées.
👉 Gardez-le **activé** avec le modèle moléculaire (56 variables seraient illisibles),
**désactivez-le** pour identifier la variable précise qui pilote la prédiction.

**Réglages de calcul**: compromis précision / rapidité / mémoire.
*Taille du fond* : combien de patients de référence définissent la baseline du
« patient moyen ». *nsamples* : combien de combinaisons de variables SHAP teste pour
estimer chaque contribution.
👉 Les valeurs par défaut (25 / 120) constituent un bon équilibre. Augmentez-les pour des
**valeurs stables, de qualité publication** ; baissez-les si l'application devient lente ou
sature la mémoire. À noter : modifier la taille du fond déplace légèrement la baseline,
puisque la population de référence change.

---
Les résultats sont **mis en cache** par profil, horizon et réglages
""",
        "xai_model_caption": "Explication du modèle **{model}** (le même que celui affiché ci-dessus), "
        "pour le profil actuellement saisi dans la barre latérale.",
        "xai_model_clin": "clinique seul",
        "xai_model_full": "clinique + moléculaire",
        "xai_horizon": "Horizon (années)",
        "xai_grouped": "Regrouper par famille",
        "xai_settings": "⚙️ Réglages de calcul (mémoire / précision)",
        "xai_bg": "Taille du fond de référence",
        "xai_ns": "nsamples",
        "xai_settings_cap": "Plus élevé = plus précis mais plus lourd. Si l'application sature la "
        "mémoire, baissez les deux. Le résultat est mis en cache par "
        "(profil, horizon, réglages).",
        "xai_button": "🔎 Expliquer cette prédiction",
        "xai_spinner": "Calcul des contributions SHAP…",
        "xai_nobg": "Fichier de référence introuvable. Exportez-le une fois depuis le notebook :\n\n"
        "`X_bg.sample(100, random_state=0).to_parquet('models/full/shap_background.parquet')`",
        "xai_m_pred": "Survie prédite à {t:g} ans",
        "xai_m_base": "Baseline (patient moyen)",
        "xai_delta": "{v:+.1f} pts vs moyenne",
        "xai_up": "Augmente la survie",
        "xai_down": "Diminue la survie",
        "xai_xlabel": "Contribution à la survie à {t:g} ans",
        "xai_read_head": "**Comment lire ce graphe :** chaque barre montre comment *la valeur de ce "
        "patient* déplace la survie **par rapport au patient moyen**, et non si la "
        "variable est bonne ou mauvaise en soi. ",
        "xai_read_gene": "Exemple : `{g} mutation : Non` en vert signifie que l'**absence** de mutation "
        "{g} augmente la survie par rapport à une population où certains patients la "
        "portent — cela ne veut *pas* dire que la mutation est protectrice.",
        "xai_read_generic": "Exemple : `{lab}` est comparé à la valeur moyenne de la cohorte "
        "d'entraînement — une barre verte signifie que la valeur de ce patient est "
        "plus favorable que cette moyenne, et non qu'elle est normale dans l'absolu.",
        "xai_sanity": "Contrôle : baseline + Σ contributions = {recon:.3f} (= survie prédite {s:.3f} ✓). "
        "🟢 hémoglobine / plaquettes élevées augmentent la survie ; 🔴 blastes élevés / "
        "cytogénétique défavorable la diminuent — cohérent avec l'analyse de Cox univariée.",
        "lab_yes": "Oui",
        "lab_no": "Non",
        "lab_mutation": "mutation {g}",
        "lab_effect": "Effet : {e}",
        "lab_cyto": "Cytogénétique (classe ELN)",
        "lab_center": "Centre de traitement",
        "err_nomodel": "Modèle introuvable. Exécutez d'abord le notebook d'entraînement.\n\n{err}",
        "err_predict": "Erreur de prédiction : {err}",
        "idle": "👈 Saisissez le profil clinique du patient et choisissez un scénario moléculaire dans "
        "la barre latérale, puis cliquez sur **Prédire la survie** pour comparer les deux "
        "modèles.",
        "load_clin": "Chargement du modèle clinique...",
        "load_full": "Chargement du modèle clinique + moléculaire...",
    },
}
