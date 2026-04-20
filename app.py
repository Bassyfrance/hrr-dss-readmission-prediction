# ============================================================
# HOSPITAL READMISSION RISK DSS (HRR-DSS)
# ============================================================

# ============================================================
# What this app is doing:
# - Provide a multi-section clinical DSS dashboard
# - Display executive overview metrics
# - Show patient population insights
# - Prepare the app structure for additional DSS sections
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="HRR-DSS", page_icon="🏥", layout="wide")

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #123C69;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        font-size: 17px;
        color: #4B6584;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #0F4C75;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .insight-box {
        background-color: #F4F8FB;
        border-left: 6px solid #3282B8;
        padding: 12px 16px;
        border-radius: 10px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.markdown('<div class="main-title">Hospital Readmission Risk Decision Support System (HRR-DSS)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An Intelligent Clinical Dashboard for Predicting Early Readmission and Supporting Discharge Decisions</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetic_data.csv")
    df = df.replace("?", np.nan)

    # Drop high-missing columns
    drop_cols = ["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Fill important missing values
    for col in ["race", "diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Remove invalid gender if present
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"]

    # Convert age to numeric safely
    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95
    }

    if "age" in df.columns:
        if df["age"].dtype == object:
            df["age"] = df["age"].map(age_map)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Create binary target safely
    if "readmitted" in df.columns:
        df["readmitted_binary"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Age band safely
    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
            include_lowest=True
        )

    return df

df = load_data()

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio(
    "Go to Section",
    [
        "Executive Overview",
        "Patient Population Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick System Information")
st.sidebar.write("**Selected Model:** Logistic Regression")
st.sidebar.write("**Target:** Early readmission (<30 days)")
st.sidebar.caption("For educational decision support use only.")

# ------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------
def executive_overview(df):
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)

    total_patients = len(df)
    high_risk = int(df["readmitted_binary"].sum())
    low_risk = int(total_patients - high_risk)
    readmit_rate = round((high_risk / total_patients) * 100, 2)
    avg_age = round(df["age"].mean(), 2)
    avg_stay = round(df["time_in_hospital"].mean(), 2)
    avg_meds = round(df["num_medications"].mean(), 2)
    avg_inpatient = round(df["number_inpatient"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patient Encounters", f"{total_patients:,}")
    c2.metric("High-Risk Patients", f"{high_risk:,}")
    c3.metric("Low-Risk Patients", f"{low_risk:,}")
    c4.metric("Readmission Rate (%)", f"{readmit_rate}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Average Age", avg_age)
    c6.metric("Average Stay (Days)", avg_stay)
    c7.metric("Average Medications", avg_meds)
    c8.metric("Average Inpatient Visits", avg_inpatient)

    st.markdown(
        '<div class="insight-box"><b>Insight:</b> This overview summarises the scale of the readmission problem and provides clinicians with a quick population-level understanding before moving into deeper analytical and predictive sections.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["readmitted_binary"].value_counts().sort_index()
        ax.bar(["Low Risk", "High Risk"], counts.values)
        ax.set_title("Readmission Class Distribution")
        ax.set_ylabel("Number of Patients")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        age_counts = df["age_band"].astype(str).value_counts().sort_index()
        ax.bar(age_counts.index, age_counts.values)
        ax.set_title("Patient Distribution by Age Band")
        ax.set_xlabel("Age Band")
        ax.set_ylabel("Number of Patients")
        st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 2: PATIENT POPULATION INSIGHTS
# ------------------------------------------------------------
def patient_population_insights(df):
    st.markdown('<div class="section-title">Patient Population Insights</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box"><b>Purpose:</b> This section explores the demographic and encounter-level characteristics of the patient population to understand how readmission patterns vary across groups.</div>',
        unsafe_allow_html=True
    )

    # -------------------------------
    # Readmission by Gender
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        gender_readmit = pd.crosstab(df["gender"], df["readmitted_binary"])
        fig, ax = plt.subplots(figsize=(6, 4))
        gender_readmit.plot(kind="bar", ax=ax)
        ax.set_title("Readmission by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        ax.legend(["Low Risk", "High Risk"])
        st.pyplot(fig)

    with col2:
        race_counts = df["race"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(6, 4))
        race_counts.plot(kind="bar", ax=ax)
        ax.set_title("Top Race Categories")
        ax.set_xlabel("Race")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    # -------------------------------
    # Age vs Readmission
    # -------------------------------
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="readmitted_binary", y="age", data=df, ax=ax)
        ax.set_title("Age vs Readmission Risk")
        ax.set_xlabel("Readmitted (0 = No, 1 = Yes)")
        ax.set_ylabel("Age")
        st.pyplot(fig)

    with col4:
        admission_readmit = pd.crosstab(df["admission_type_id"], df["readmitted_binary"])
        fig, ax = plt.subplots(figsize=(6, 4))
        admission_readmit.plot(kind="bar", ax=ax)
        ax.set_title("Admission Type vs Readmission")
        ax.set_xlabel("Admission Type ID")
        ax.set_ylabel("Count")
        ax.legend(["Low Risk", "High Risk"])
        st.pyplot(fig)

    # -------------------------------
    # Discharge Disposition vs Readmission
    # -------------------------------
    st.markdown("#### Discharge Disposition and Readmission")

    top_discharge = df["discharge_disposition_id"].value_counts().head(10).index
    discharge_df = df[df["discharge_disposition_id"].isin(top_discharge)]
    discharge_readmit = pd.crosstab(discharge_df["discharge_disposition_id"], discharge_df["readmitted_binary"])

    fig, ax = plt.subplots(figsize=(10, 5))
    discharge_readmit.plot(kind="bar", ax=ax)
    ax.set_title("Top Discharge Disposition IDs vs Readmission")
    ax.set_xlabel("Discharge Disposition ID")
    ax.set_ylabel("Count")
    ax.legend(["Low Risk", "High Risk"])
    st.pyplot(fig)

    st.markdown(
        '<div class="insight-box"><b>Interpretation:</b> This section helps identify which demographic and encounter groups appear more represented among high-risk patients, providing useful context for clinicians before patient-level prediction.</div>',
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# ROUTING
# ------------------------------------------------------------
if page == "Executive Overview":
    executive_overview(df)
elif page == "Patient Population Insights":
    patient_population_insights(df)
