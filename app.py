# ============================================================
# HOSPITAL READMISSION RISK DSS (HRR-DSS)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="HRR-DSS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #F8FBFF, #EEF4FA);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
}
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #123C69;
    margin-bottom: 0.15rem;
    letter-spacing: -0.5px;
}
.sub-title {
    font-size: 18px;
    color: #486581;
    margin-bottom: 0.8rem;
}
.header-banner {
    background: linear-gradient(135deg, #EAF4FF, #F8FBFF);
    border: 1px solid #D7E6F5;
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #0F4C75;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}
.insight-box {
    background: #F4F8FB;
    border-left: 6px solid #3282B8;
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.card-box {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 3px 14px rgba(0,0,0,0.05);
    margin-bottom: 0.8rem;
}
.page-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #CAD9E8, transparent);
    margin-top: 0.6rem;
    margin-bottom: 1rem;
}
.risk-low {
    background: #E8F5E9;
    border-left: 6px solid #2E7D32;
    padding: 16px;
    border-radius: 12px;
    color: #1B5E20;
    margin-top: 0.8rem;
}
.risk-medium {
    background: #FFF8E1;
    border-left: 6px solid #F9A825;
    padding: 16px;
    border-radius: 12px;
    color: #8D6E00;
    margin-top: 0.8rem;
}
.risk-high {
    background: #FFEBEE;
    border-left: 6px solid #C62828;
    padding: 16px;
    border-radius: 12px;
    color: #8E0000;
    margin-top: 0.8rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2747 0%, #163A63 100%);
    border-right: 1px solid #294E78;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.sidebar-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-bottom: 0.3rem;
}
.sidebar-subtitle {
    font-size: 13px;
    color: #D8E7F5;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #EFF6FB;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #D7E9F8 !important;
    color: #123C69 !important;
}
.footer-box {
    margin-top: 1.5rem;
    padding: 14px 16px;
    border-radius: 14px;
    background: #F7FAFC;
    border: 1px solid #E2E8F0;
    color: #51606F;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="main-title">🏥 Hospital Readmission Risk Decision Support System (HRR-DSS)</div>
    <div class="sub-title">An Intelligent Clinical Dashboard for Predicting Early Readmission and Supporting Discharge Decisions</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_boxplot(data, x_col, y_col, title, xlabel=None, ylabel=None, map_labels=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = data[[x_col, y_col]].copy() if {x_col, y_col}.issubset(data.columns) else pd.DataFrame()

    if not plot_df.empty:
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col, y_col])

        if map_labels:
            plot_df[x_col] = plot_df[x_col].map(map_labels)

        plot_df = plot_df.dropna(subset=[x_col])

        if not plot_df.empty and plot_df[x_col].nunique() >= 1:
            sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax)
        else:
            ax.text(0.5, 0.5, "No valid data available", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Required columns missing", ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel if ylabel else y_col)
    st.pyplot(fig)
    plt.close(fig)


def safe_bar_from_series(series, title, xlabel="", ylabel="Count", rotate=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    if series is not None and len(series) > 0:
        series.plot(kind="bar", ax=ax)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)


def safe_bar_from_crosstab(ctab, title, xlabel="", ylabel="Count", legend_labels=None, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if ctab is not None and not ctab.empty:
        ctab.plot(kind="bar", ax=ax)
        if legend_labels is not None:
            ax.legend(legend_labels)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetic_data.csv")
    df = df.replace("?", np.nan)

    drop_cols = ["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in ["race", "diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"].copy()

    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }

    if "age" in df.columns:
        df["age"] = df["age"].astype(str).str.strip()
        df["age"] = df["age"].replace({"nan": np.nan})
        df["age"] = df["age"].map(age_map)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "readmitted" in df.columns:
        df["readmitted_binary"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
        df["readmitted_binary"] = pd.to_numeric(df["readmitted_binary"], errors="coerce")

    numeric_cols = [
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [c for c in ["age", "readmitted_binary"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols).copy()

    if "readmitted_binary" in df.columns:
        df["readmitted_binary"] = df["readmitted_binary"].astype(int)

    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
            include_lowest=True
        )

    return df


df = load_data()

SELECTED_FEATURES = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "insulin", "change", "diabetesMed"
]

# Keep only usable rows for modeling
model_df = df.copy()
model_df = model_df.dropna(subset=[c for c in SELECTED_FEATURES + ["readmitted_binary"] if c in model_df.columns]).copy()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-title">🧭 HRR-DSS</div>
<div class="sidebar-subtitle">Clinical Intelligence Dashboard</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-card">
    <b>System Purpose</b><br>
    Support early readmission prediction, discharge planning, and clinical decision-making.
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Population Insights",
        "Risk Drivers",
        "Prediction Engine",
        "Clinical Recommendation Engine",
        "Model Evaluation and Trust Dashboard"
    ]
)

st.sidebar.markdown("""
<div class="sidebar-card">
    <b>Global Filters</b><br>
    Refine dashboard views by selected profile.
</div>
""", unsafe_allow_html=True)

gender_options = ["All"] + sorted(df["gender"].dropna().astype(str).unique().tolist()) if "gender" in df.columns else ["All"]
race_options = ["All"] + sorted(df["race"].dropna().astype(str).unique().tolist()) if "race" in df.columns else ["All"]
ageband_options = ["All"] + sorted(df["age_band"].dropna().astype(str).unique().tolist()) if "age_band" in df.columns else ["All"]

selected_gender_filter = st.sidebar.selectbox("Filter by Gender", gender_options)
selected_race_filter = st.sidebar.selectbox("Filter by Race", race_options)
selected_age_band_filter = st.sidebar.selectbox("Filter by Age Band", ageband_options)

st.sidebar.markdown("""
<div class="sidebar-card">
    <b>Selected Model</b><br>
    Logistic Regression<br><br>
    <b>Target</b><br>
    Early readmission (&lt;30 days)
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-card">
    <b>Usage Note</b><br>
    This DSS is for educational and analytical decision support purposes.
</div>
""", unsafe_allow_html=True)

filtered_df = df.copy()
if selected_gender_filter != "All" and "gender" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender_filter]
if selected_race_filter != "All" and "race" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["race"] == selected_race_filter]
if selected_age_band_filter != "All" and "age_band" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["age_band"].astype(str) == selected_age_band_filter]

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------
@st.cache_resource
def train_selected_model(train_df):
    X = train_df[SELECTED_FEATURES].copy()
    y = train_df["readmitted_binary"].copy()

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    model.fit(X, y)
    return model


@st.cache_resource
def evaluate_models(train_df):
    X = train_df[SELECTED_FEATURES].copy()
    y = train_df["readmitted_binary"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    def make_preprocessor():
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ]
        )

    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Decision Tree": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", GradientBoostingClassifier(random_state=42))
        ])
    }

    records = []
    predictions = {}

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        y_prob = mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, "predict_proba") else None

        records.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        })

        predictions[name] = {"model": mdl, "y_pred": y_pred, "y_prob": y_prob}

    results_df = pd.DataFrame(records).sort_values(by="Recall", ascending=False)
    return results_df, predictions, y_test


model = train_selected_model(model_df)
results_df, all_predictions, y_test_eval = evaluate_models(model_df)

# ------------------------------------------------------------
# PREDICTION FORM
# ------------------------------------------------------------
def prediction_form():
    c1, c2, c3 = st.columns(3)

    with c1:
        race = st.selectbox("Race", sorted(model_df["race"].dropna().astype(str).unique().tolist()))
        gender = st.selectbox("Gender", sorted(model_df["gender"].dropna().astype(str).unique().tolist()))
        age = st.slider("Age", 0, 100, 65)
        admission_type_id = st.selectbox("Admission Type ID", sorted(model_df["admission_type_id"].dropna().astype(int).unique().tolist()))
        discharge_disposition_id = st.selectbox("Discharge Disposition ID", sorted(model_df["discharge_disposition_id"].dropna().astype(int).unique().tolist()))
        admission_source_id = st.selectbox("Admission Source ID", sorted(model_df["admission_source_id"].dropna().astype(int).unique().tolist()))

    with c2:
        time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 44)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 81, 15)
        number_outpatient = st.slider("Number of Outpatient Visits", 0, 42, 0)
        number_emergency = st.slider("Number of Emergency Visits", 0, 76, 0)

    with c3:
        number_inpatient = st.slider("Number of Inpatient Visits", 0, 21, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 8)
        insulin = st.selectbox("Insulin", sorted(model_df["insulin"].dropna().astype(str).unique().tolist()))
        change = st.selectbox("Medication Change", sorted(model_df["change"].dropna().astype(str).unique().tolist()))
        diabetesMed = st.selectbox("Diabetes Medication", sorted(model_df["diabetesMed"].dropna().astype(str).unique().tolist()))

    return pd.DataFrame([{
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed
    }])


def interpret_risk(risk_prob):
    if risk_prob < 0.30:
        return {
            "risk_level": "Low Risk",
            "css_class": "risk-low",
            "urgency": "Routine",
            "recommendation": "Proceed with standard discharge plan, routine follow-up, and standard patient education."
        }
    elif risk_prob < 0.60:
        return {
            "risk_level": "Moderate Risk",
            "css_class": "risk-medium",
            "urgency": "Enhanced Monitoring",
            "recommendation": "Recommend enhanced discharge review, medication counselling, and follow-up contact within 7–14 days."
        }
    else:
        return {
            "risk_level": "High Risk",
            "css_class": "risk-high",
            "urgency": "Urgent Follow-Up",
            "recommendation": "Recommend urgent care coordination, medication reconciliation, and post-discharge follow-up within 7 days."
        }

# ------------------------------------------------------------
# PAGE 1
# ------------------------------------------------------------
def overview():
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    total = len(filtered_df)
    high = int(filtered_df["readmitted_binary"].sum()) if "readmitted_binary" in filtered_df.columns else 0
    low = total - high
    rate = round((high / total) * 100, 2) if total > 0 else 0
    avg_age = round(filtered_df["age"].dropna().mean(), 2) if total > 0 and "age" in filtered_df.columns else 0
    avg_stay = round(filtered_df["time_in_hospital"].dropna().mean(), 2) if total > 0 and "time_in_hospital" in filtered_df.columns else 0
    avg_meds = round(filtered_df["num_medications"].dropna().mean(), 2) if total > 0 and "num_medications" in filtered_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{total:,}")
    c2.metric("High Risk", f"{high:,}")
    c3.metric("Low Risk", f"{low:,}")
    c4.metric("Readmission Rate", f"{rate}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Average Age", avg_age)
    c6.metric("Average Stay", avg_stay)
    c7.metric("Average Medications", avg_meds)

    st.markdown(
        '<div class="insight-box"><b>Insight:</b> This overview summarises the filtered patient population and highlights the scale of early hospital readmission risk.</div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📊 Summary Charts", "📌 Interpretation"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            counts = filtered_df["readmitted_binary"].value_counts().sort_index() if "readmitted_binary" in filtered_df.columns else pd.Series(dtype=float)
            counts = pd.Series([counts.get(0, 0), counts.get(1, 0)], index=["Low Risk", "High Risk"])
            safe_bar_from_series(counts, "Readmission Distribution", ylabel="Count")

        with col2:
            age_counts = filtered_df["age_band"].astype(str).value_counts().sort_index() if "age_band" in filtered_df.columns else pd.Series(dtype=float)
            safe_bar_from_series(age_counts, "Patient Distribution by Age Band", ylabel="Count")

    with tab2:
        st.write("This section provides a high-level dashboard view for quick situational awareness.")
        st.write("Use the global filters in the sidebar to refine the patient cohort and review how top-level indicators change.")

# ------------------------------------------------------------
# PAGE 2
# ------------------------------------------------------------
def population():
    st.markdown('<div class="section-title">Population Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box"><b>Purpose:</b> Understand how readmission varies across patient groups.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👥 Demographics", "🏥 Encounter Patterns"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            gender_ct = pd.crosstab(filtered_df["gender"], filtered_df["readmitted_binary"]) if {"gender", "readmitted_binary"}.issubset(filtered_df.columns) else pd.DataFrame()
            if not gender_ct.empty:
                gender_ct = gender_ct.reindex(columns=[0, 1], fill_value=0)
            safe_bar_from_crosstab(gender_ct, "Readmission by Gender", ylabel="Count", legend_labels=["Low Risk", "High Risk"])

        with col2:
            safe_boxplot(
                filtered_df,
                x_col="readmitted_binary",
                y_col="age",
                title="Age vs Readmission",
                xlabel="Readmission Group",
                ylabel="Age",
                map_labels={0: "Low Risk", 1: "High Risk"}
            )

    with tab2:
        col3, col4 = st.columns(2)

        with col3:
            adm_ct = pd.crosstab(filtered_df["admission_type_id"], filtered_df["readmitted_binary"]) if {"admission_type_id", "readmitted_binary"}.issubset(filtered_df.columns) else pd.DataFrame()
            if not adm_ct.empty:
                adm_ct = adm_ct.reindex(columns=[0, 1], fill_value=0)
            safe_bar_from_crosstab(adm_ct, "Admission Type vs Readmission", legend_labels=["Low Risk", "High Risk"])

        with col4:
            top_races = filtered_df["race"].value_counts().head(8) if "race" in filtered_df.columns else pd.Series(dtype=float)
            safe_bar_from_series(top_races, "Top Race Categories", ylabel="Count", rotate=True)

# ------------------------------------------------------------
# PAGE 3
# ------------------------------------------------------------
def risk_drivers():
    st.markdown('<div class="section-title">Risk Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box"><b>Goal:</b> Identify key factors that influence readmission risk.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Utilisation Signals", "🧪 Clinical Complexity"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            safe_boxplot(
                filtered_df,
                x_col="readmitted_binary",
                y_col="number_inpatient",
                title="Inpatient Visits vs Readmission",
                xlabel="Readmission Group",
                ylabel="Number of Inpatient Visits",
                map_labels={0: "Low Risk", 1: "High Risk"}
            )

        with col2:
            safe_boxplot(
                filtered_df,
                x_col="readmitted_binary",
                y_col="number_emergency",
                title="Emergency Visits vs Readmission",
                xlabel="Readmission Group",
                ylabel="Number of Emergency Visits",
                map_labels={0: "Low Risk", 1: "High Risk"}
            )

    with tab2:
        col3, col4 = st.columns(2)

        with col3:
            safe_boxplot(
                filtered_df,
                x_col="readmitted_binary",
                y_col="time_in_hospital",
                title="Length of Stay vs Readmission",
                xlabel="Readmission Group",
                ylabel="Time in Hospital",
                map_labels={0: "Low Risk", 1: "High Risk"}
            )

        with col4:
            safe_boxplot(
                filtered_df,
                x_col="readmitted_binary",
                y_col="num_medications",
                title="Medications vs Readmission",
                xlabel="Readmission Group",
                ylabel="Number of Medications",
                map_labels={0: "Low Risk", 1: "High Risk"}
            )

# ------------------------------------------------------------
# PAGE 4
# ------------------------------------------------------------
def prediction_engine():
    st.markdown('<div class="section-title">🧠 Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Purpose:</b> Enter patient clinical data to estimate the probability of early hospital readmission.
    This module simulates an AI-assisted clinical decision process.
    </div>
    """, unsafe_allow_html=True)

    input_df = prediction_form()

    st.markdown("### 🧾 Patient Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Run AI Readmission Risk Prediction"):
        risk_prob = model.predict_proba(input_df)[0][1]
        risk_class = model.predict(input_df)[0]
        info = interpret_risk(risk_prob)

        st.markdown("## Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="card-box"><h4>Risk Probability</h4><h2>{risk_prob:.2%}</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card-box"><h4>Risk Class</h4><h2>{"High Risk" if risk_class == 1 else "Low Risk"}</h2></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card-box"><h4>Risk Level</h4><h2>{info["risk_level"]}</h2></div>', unsafe_allow_html=True)

        st.markdown("### Risk Level Indicator")
        st.progress(int(risk_prob * 100))

        st.markdown(
            f'<div class="{info["css_class"]}"><b>{info["risk_level"]}</b><br>'
            f'Estimated probability of early readmission: {risk_prob:.2%}</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Clinical Interpretation")
        if risk_prob < 0.30:
            st.success("Patient is considered stable with low likelihood of early readmission.")
        elif risk_prob < 0.60:
            st.warning("Patient shows moderate risk. Additional monitoring is recommended.")
        else:
            st.error("Patient is at high risk of early readmission. Immediate intervention is advised.")

# ------------------------------------------------------------
# PAGE 5
# ------------------------------------------------------------
def clinical_recommendation_engine():
    st.markdown('<div class="section-title">Clinical Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Purpose:</b> Convert predicted readmission risk into clear, practical, and clinically meaningful discharge actions.
    </div>
    """, unsafe_allow_html=True)

    input_df = prediction_form()

    st.markdown("### Patient Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Generate Clinical Recommendation"):
        risk_prob = model.predict_proba(input_df)[0][1]
        info = interpret_risk(risk_prob)

        st.markdown("## Decision Support Output")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="card-box"><h4>Risk Probability</h4><h2>{risk_prob:.2%}</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card-box"><h4>Risk Category</h4><h2>{info["risk_level"]}</h2></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card-box"><h4>Urgency Level</h4><h2>{info["urgency"]}</h2></div>', unsafe_allow_html=True)

        st.progress(int(risk_prob * 100))
        st.markdown(f'<div class="{info["css_class"]}"><b>{info["risk_level"]}</b><br>{info["recommendation"]}</div>', unsafe_allow_html=True)

        if risk_prob < 0.30:
            actions = [
                "Proceed with standard discharge planning.",
                "Provide routine discharge instructions.",
                "Encourage medication adherence and self-monitoring.",
                "Schedule normal follow-up according to standard care pathway."
            ]
        elif risk_prob < 0.60:
            actions = [
                "Review discharge readiness before release.",
                "Provide enhanced medication counselling.",
                "Arrange follow-up contact within 7–14 days.",
                "Consider patient education on warning signs and return precautions."
            ]
        else:
            actions = [
                "Initiate urgent discharge planning review.",
                "Arrange follow-up within 7 days.",
                "Conduct medication reconciliation before discharge.",
                "Escalate to care coordination / case management team.",
                "Flag patient for enhanced post-discharge monitoring."
            ]

        st.markdown("### Recommended Clinical Actions")
        for action in actions:
            st.write(f"- {action}")

# ------------------------------------------------------------
# PAGE 6
# ------------------------------------------------------------
def trust_dashboard():
    st.markdown('<div class="section-title">Model Evaluation and Trust Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box"><b>Purpose:</b> Show why the chosen prediction model can be trusted for decision support.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Model Table", "Visual Comparison", "Final Justification"])

    with tab1:
        st.dataframe(results_df.round(4), use_container_width=True)

    with tab2:
        st.markdown("#### Performance Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Confusion Matrix — Selected Model")
        y_pred_lr = all_predictions["Logistic Regression"]["y_pred"]
        cm = confusion_matrix(y_test_eval, y_pred_lr)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, pred in all_predictions.items():
            if pred["y_prob"] is not None:
                fpr, tpr, _ = roc_curve(y_test_eval, pred["y_prob"])
                ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown(
            '<div class="insight-box"><b>Final Model Decision:</b> Logistic Regression was selected because it achieved the strongest recall for high-risk patients. In a healthcare DSS, correctly identifying high-risk patients is more important than maximising accuracy alone.</div>',
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# ROUTING
# ------------------------------------------------------------
if page == "Executive Overview":
    overview()
elif page == "Population Insights":
    population()
elif page == "Risk Drivers":
    risk_drivers()
elif page == "Prediction Engine":
    prediction_engine()
elif page == "Clinical Recommendation Engine":
    clinical_recommendation_engine()
else:
    trust_dashboard()

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer-box">
<b>HRR-DSS</b> · Hospital Readmission Risk Decision Support System<br>
Designed to support early readmission analysis, predictive assessment, and discharge-related decision support in a clinical context.
</div>
""", unsafe_allow_html=True)
