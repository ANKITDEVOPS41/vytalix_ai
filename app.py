import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import uuid
import plotly.express as px

st.set_page_config(
    page_title="Vytalix AI",
    page_icon="🫀",
    layout="wide"
)

# =========================
# GLOBAL WHITE-BLUE THEME
# =========================
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(to bottom right, #F7FCFF, #EAF6FF) !important;
    color: #12344D !important;
}

/* Main container */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #F7FCFF, #EAF6FF) !important;
}

/* Header area */
[data-testid="stHeader"] {
    background: #F7FCFF !important;
}

/* Toolbar */
[data-testid="stToolbar"] {
    right: 1rem;
}

/* Main block */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #DFF1FF, #CFEAFF) !important;
    border-right: 1px solid #B9DFFF !important;
}

section[data-testid="stSidebar"] * {
    color: #0B3C6D !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #0A3D62 !important;
    font-weight: 700 !important;
}

/* General text */
p, label, span, div {
    color: #1F2937;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #12344D !important;
    border: 1px solid #CFE8FF !important;
    border-radius: 10px !important;
}

/* Selectbox main */
.stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #12344D !important;
    border: 1px solid #CFE8FF !important;
    border-radius: 10px !important;
}

/* Stronger dropdown popup fixes */
div[data-baseweb="popover"] {
    background: #FFFFFF !important;
    color: #0B3C6D !important;
}

div[role="listbox"] {
    background: #FFFFFF !important;
    color: #0B3C6D !important;
    border: 1px solid #CFE8FF !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 18px rgba(45,140,255,0.12) !important;
}

div[role="option"] {
    background: #FFFFFF !important;
    color: #0B3C6D !important;
}

div[role="option"]:hover {
    background: #DFF1FF !important;
    color: #0B3C6D !important;
}

li[role="option"] {
    background: #FFFFFF !important;
    color: #0B3C6D !important;
}

li[role="option"]:hover {
    background: #DFF1FF !important;
    color: #0B3C6D !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4DA8FF, #2D8CFF) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(45,140,255,0.18) !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2D8CFF, #1976D2) !important;
    color: white !important;
}

/* Tables */
[data-testid="stTable"] {
    background: #FFFFFF !important;
    border: 1px solid #D7ECFF !important;
    border-radius: 16px !important;
    padding: 6px !important;
}

table {
    background: #FFFFFF !important;
    color: #12344D !important;
    border-collapse: collapse !important;
}

thead tr th {
    background: #DFF1FF !important;
    color: #0B3C6D !important;
}

tbody tr td {
    background: #FFFFFF !important;
    color: #12344D !important;
}

/* Markdown code blocks */
code, pre {
    background: #EAF6FF !important;
    color: #0B3C6D !important;
    border-radius: 10px !important;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 14px !important;
}

/* White card look */
.white-card {
    background: #FFFFFF !important;
    border: 1px solid #D7ECFF !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 18px rgba(45,140,255,0.10) !important;
    padding: 18px !important;
}

/* Tabs / expanders fallback */
.streamlit-expanderHeader {
    background: #EAF6FF !important;
    color: #0B3C6D !important;
}

/* Smooth markdown sections */
.element-container {
    border-radius: 12px;
}

/* Badge */
.custom-badge {
    display:inline-block;
    background:#DFF1FF;
    color:#0B3C6D;
    padding:10px 16px;
    border-radius:999px;
    border:1px solid #B9DFFF;
    font-weight:600;
    font-size:14px;
}

/* Plotly chart container */
.js-plotly-plot, .plotly, .plot-container {
    background: #FFFFFF !important;
}

/* Plotly modebar */
.modebar {
    background: rgba(223, 241, 255, 0.95) !important;
    border-radius: 10px !important;
}
.modebar-btn svg {
    fill: #0B3C6D !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    color: #2D8CFF !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load model assets
# =========================
@st.cache_resource
def load_assets():
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


model, scaler, feature_names = load_assets()

# =========================
# Session state
# =========================
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

# =========================
# Human-readable mappings
# =========================
SEX_MAP = {"Female": 0, "Male": 1}

CHEST_PAIN_MAP = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-Anginal Pain": 3,
    "Asymptomatic": 4
}

FBS_MAP = {"No": 0, "Yes": 1}

EKG_MAP = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

ANGINA_MAP = {"No": 0, "Yes": 1}

SLOPE_MAP = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}

THAL_MAP = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

# =========================
# UI Helper Functions
# =========================
def metric_card(title, value, color):
    st.markdown(
        f"""
        <div style="
            background:{color};
            padding:18px;
            border-radius:16px;
            color:white;
            text-align:center;
            box-shadow:0 6px 16px rgba(0,0,0,0.10);
            min-height:100px;
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <div style="font-size:15px; opacity:0.95;">{title}</div>
            <div style="font-size:24px; font-weight:700; margin-top:6px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_box(title, content, bg="#F4FAFF", border="#CFE8FF"):
    st.markdown(
        f"""
        <div style="
            background:{bg};
            padding:18px;
            border-radius:16px;
            border:1px solid {border};
            box-shadow:0 4px 14px rgba(45,140,255,0.08);
            margin-bottom:12px;
            color:#0B1F3A;
        ">
            <h4 style="margin-top:0; color:#0B3C6D;">{title}</h4>
            <div style="color:#1F2937; font-size:16px; line-height:1.65;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Core logic
# =========================
def determine_risk(pred_prob, spo2, temperature, rr, bp, chest_pain, exercise_angina):
    if spo2 < 90 or temperature >= 102.0 or rr >= 28:
        return "HIGH RISK", "#D64545"

    if bp >= 180:
        return "HIGH RISK", "#D64545"

    if exercise_angina == 1 and chest_pain in [1, 2] and pred_prob >= 0.45:
        return "HIGH RISK", "#D64545"

    if pred_prob >= 0.65:
        return "HIGH RISK", "#D64545"
    elif pred_prob >= 0.35:
        return "MODERATE RISK", "#F39C12"
    else:
        return "LOW RISK", "#2E8B57"


def get_confidence_band(prob):
    if prob >= 0.80:
        return "High Confidence"
    elif prob >= 0.60:
        return "Moderate Confidence"
    return "Early Signal"


def get_hospital_recommendation(risk_label, spo2, temperature, rr, bp):
    if risk_label == "HIGH RISK":
        return {
            "decision": "GO TO HOSPITAL / URGENT CARE NOW",
            "color": "#D64545",
            "reason": "Critical physiological indicators and elevated risk suggest immediate medical evaluation."
        }

    if spo2 <= 92 or temperature >= 101.5 or rr >= 24 or bp >= 160:
        return {
            "decision": "CLINICAL REVIEW SHOULD NOT BE DELAYED",
            "color": "#F39C12",
            "reason": "Worsening indicators suggest that a hospital or physician visit is advisable soon."
        }

    if risk_label == "MODERATE RISK":
        return {
            "decision": "SCHEDULE CLINICAL CONSULTATION",
            "color": "#F39C12",
            "reason": "The patient should seek medical review if symptoms persist or worsen."
        }

    return {
        "decision": "HOME MONITORING IS APPROPRIATE",
        "color": "#2E8B57",
        "reason": "Current findings do not indicate immediate hospital referral, but continued monitoring is recommended."
    }


def get_escalation_status(risk_label):
    if risk_label == "LOW RISK":
        return "No escalation required"
    elif risk_label == "MODERATE RISK":
        return "Clinical review recommended"
    return "Urgent escalation required"


def recommendation_text(risk_label, mode):
    if mode == "Personal Assist":
        if risk_label == "LOW RISK":
            return "Continue home monitoring and preventive care."
        elif risk_label == "MODERATE RISK":
            return "Consult a clinician and continue rechecking vitals."
        return "Seek urgent medical care immediately."

    elif mode == "Clinic Assist":
        if risk_label == "LOW RISK":
            return "Low triage priority. Continue observation and preventive guidance."
        elif risk_label == "MODERATE RISK":
            return "Moderate triage priority. Structured clinical review recommended."
        return "High triage priority. Immediate physician review recommended."

    else:
        if risk_label == "LOW RISK":
            return "Combined historical and current data support continued monitoring."
        elif risk_label == "MODERATE RISK":
            return "Combined signals suggest clinical follow-up and closer observation."
        return "Combined wearable and clinical signals indicate urgent escalation."


def preventive_suggestions(risk_label):
    if risk_label == "LOW RISK":
        return [
            "Maintain hydration and rest appropriately.",
            "Continue periodic monitoring of vitals.",
            "Recheck symptoms within 12–24 hours if needed."
        ]
    elif risk_label == "MODERATE RISK":
        return [
            "Reduce physical exertion and rest.",
            "Recheck vitals in the next 4–6 hours.",
            "Schedule a clinician consultation.",
            "Seek earlier review if symptoms worsen."
        ]
    else:
        return [
            "Do not delay physician review.",
            "Nearest healthcare response should be considered.",
            "Avoid physical exertion until clinical assessment.",
            "Emergency escalation may be necessary."
        ]


def build_full_input_dict(
    age,
    sex,
    chest_pain,
    bp,
    cholesterol,
    fbs,
    ekg,
    max_hr,
    exercise_angina,
    st_depression,
    slope,
    vessels,
    thallium,
    spo2,
    temperature,
    rr,
    symptom_severity,
    activity_fatigue
):
    return {
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thallium,
        "SpO2": spo2,
        "Temperature_F": temperature,
        "Respiratory_Rate": rr,
        "Symptom_Severity_Score": symptom_severity,
        "Activity_Fatigue_Score": activity_fatigue
    }


def explain_prediction(input_df, feature_importances):
    row = input_df.iloc[0]
    contributions = {}
    for feature, importance in zip(input_df.columns, feature_importances):
        contributions[feature] = abs(row[feature]) * importance

    top_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]
    result = pd.DataFrame(top_features, columns=["Feature", "Relative Influence"])
    result["Relative Influence"] = result["Relative Influence"].round(3)
    return result


def user_friendly_summary(mode, age, sex_label, spo2, temperature, rr, symptom_severity, activity_fatigue):
    summary = {
        "Mode": mode,
        "Age": age,
        "Sex": sex_label,
        "SpO₂": spo2,
        "Temperature (°F)": temperature,
        "Respiratory Rate": rr,
        "Symptom Severity": symptom_severity,
        "Activity/Fatigue": activity_fatigue
    }
    return pd.DataFrame([summary])


def generate_assessment_id():
    return f"VTX-{str(uuid.uuid4())[:8].upper()}"


def get_plain_english_risk_drivers(explanation_df):
    mapping = {
        "SpO2": "Low oxygen saturation",
        "BP": "Elevated blood pressure",
        "Symptom_Severity_Score": "High symptom severity",
        "Respiratory_Rate": "High respiratory rate",
        "Temperature_F": "Elevated temperature",
        "Cholesterol": "Elevated cholesterol",
        "Max HR": "Abnormal heart rate pattern",
        "Exercise angina": "Exertion-related chest discomfort",
        "Chest pain type": "Concerning chest pain pattern",
        "EKG results": "Abnormal ECG findings",
        "ST depression": "Cardiac stress abnormality",
        "Activity_Fatigue_Score": "Increased fatigue level"
    }

    readable = []
    for feature in explanation_df["Feature"].head(3):
        readable.append(mapping.get(feature, feature))
    return readable


def show_alert_banner(risk_label):
    if risk_label == "LOW RISK":
        st.success("🟢 Patient Stable")
    elif risk_label == "MODERATE RISK":
        st.warning("🟠 Review Recommended")
    else:
        st.error("🔴 Urgent Attention Needed")


def show_emergency_escalation(risk_label):
    if risk_label == "HIGH RISK":
        st.error(
            """
            🚨 **Emergency Escalation Triggered**  
            Nearest clinical response recommended.  
            Immediate physician review required.
            """
        )


def add_to_history(assessment_id, mode, risk_label, prob, decision):
    new_entry = {
        "Time": datetime.now().strftime("%I:%M %p"),
        "Assessment ID": assessment_id,
        "Mode": mode,
        "Risk": risk_label,
        "Probability": f"{prob * 100:.1f}%",
        "Disposition": decision
    }
    st.session_state.assessment_history.append(new_entry)

    if len(st.session_state.assessment_history) > 5:
        st.session_state.assessment_history = st.session_state.assessment_history[-5:]


def render_shareable_summary(assessment_id, risk_label, spo2, temperature, rr, recommendation):
    st.markdown("### Shareable Patient Summary")
    st.button("Generate Shareable Health Summary", disabled=True)

    st.markdown(
        f"""
        <div style="
            border:2px dashed #6FA8DC;
            border-radius:16px;
            padding:20px;
            background:#FFFFFF;
            margin-top:10px;
            color:#0B1F3A;
            box-shadow:0 4px 14px rgba(45,127,249,0.08);
        ">
            <h4 style="margin-top:0; color:#0B1F3A;">QR-Ready Patient Handoff Summary</h4>
            <p style="color:#1F2937;"><b>Patient ID:</b> {assessment_id}</p>
            <p style="color:#1F2937;"><b>Risk Level:</b> {risk_label}</p>
            <p style="color:#1F2937;"><b>Key Vitals:</b> SpO₂ {spo2}%, Temperature {temperature}°F, Respiratory Rate {rr}</p>
            <p style="color:#1F2937;"><b>Recommendation:</b> {recommendation}</p>
            <p style="color:#1F2937;"><b>Timestamp:</b> {datetime.now().strftime("%Y-%m-%d %I:%M %p")}</p>
            <div style="
                width:120px;
                height:120px;
                border:1px solid #9fbad0;
                display:flex;
                align-items:center;
                justify-content:center;
                margin-top:12px;
                background:#F9FCFF;
                font-size:14px;
                color:#666;
                border-radius:10px;
            ">
                QR Placeholder
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Header
# =========================
st.markdown(
    """
    <h1 style='text-align:center; color:#0B3C6D;'>Vytalix AI</h1>
    <p style='text-align:center; font-size:18px; color:#4F6D8A;'>
    A preventive healthcare decision-support prototype for patient monitoring,
    risk prediction, and timely clinical action.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:center; margin-bottom:14px;'>
        <span class='custom-badge'>🌐 Designed for Low-Connectivity Environments</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# Sidebar
# =========================
st.sidebar.header("Operational Mode")
mode = st.sidebar.radio(
    "Choose Mode",
    ["Personal Assist", "Clinic Assist", "Integrated Assist"]
)

st.sidebar.markdown("### Prototype Note")
st.sidebar.info(
    "This is an AI-assisted preventive decision-support prototype. "
    "It supports monitoring and triage, and does not replace clinical diagnosis."
)

# =========================
# Default clinical-neutral values
# =========================
default_chest_pain = 3
default_bp = 130
default_cholesterol = 220
default_fbs = 0
default_ekg = 0
default_max_hr = 150
default_exercise_angina = 0
default_st_depression = 1.0
default_slope = 2
default_vessels = 0
default_thallium = 3

# =========================
# Input Forms
# =========================
st.subheader(f"Input Panel — {mode}")

if mode == "Personal Assist":
    st.caption("Simple user-facing form using understandable health and symptom inputs.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 20, 100, 45)
        sex_label = st.selectbox("Sex", list(SEX_MAP.keys()))
        spo2 = st.slider("SpO₂", 80, 100, 97)

    with c2:
        temperature = st.slider("Temperature (°F)", 95.0, 105.0, 98.6, step=0.1)
        rr = st.slider("Respiratory Rate", 8, 35, 16)
        symptom_severity = st.slider("Symptom Severity Score", 1.0, 10.0, 4.0, step=0.1)

    with c3:
        activity_fatigue = st.slider("Activity / Fatigue Score", 1.0, 10.0, 5.0, step=0.1)
        chest_discomfort_label = st.selectbox("Do you feel chest discomfort?", ["No", "Yes"])
        breathlessness_label = st.selectbox("Do you feel breathlessness during activity?", ["No", "Yes"])

    sex = SEX_MAP[sex_label]
    chest_discomfort = 1 if chest_discomfort_label == "Yes" else 0
    breathlessness = 1 if breathlessness_label == "Yes" else 0

    chest_pain = 2 if chest_discomfort == 1 else default_chest_pain
    bp = 135 if symptom_severity >= 7 else default_bp
    cholesterol = default_cholesterol
    fbs = default_fbs
    ekg = default_ekg
    max_hr = 160 if activity_fatigue >= 8 else (145 if activity_fatigue >= 6 else default_max_hr)
    exercise_angina = breathlessness
    st_depression = 1.8 if breathlessness == 1 else default_st_depression
    slope = default_slope
    vessels = default_vessels
    thallium = default_thallium

elif mode == "Clinic Assist":
    st.caption("Advanced clinician-facing form using structured clinical inputs.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 20, 100, 50)
        sex_label = st.selectbox("Sex", list(SEX_MAP.keys()))
        chest_pain_label = st.selectbox("Chest Pain Type", list(CHEST_PAIN_MAP.keys()))
        bp = st.slider("Resting Blood Pressure", 80, 220, 130)
        cholesterol = st.slider("Cholesterol", 100, 600, 240)

    with c2:
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(FBS_MAP.keys()))
        ekg_label = st.selectbox("EKG Results", list(EKG_MAP.keys()))
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina_label = st.selectbox("Exercise Angina", list(ANGINA_MAP.keys()))
        st_depression = st.slider("ST Depression", 0.0, 6.5, 1.0, step=0.1)

    with c3:
        slope_label = st.selectbox("Slope of ST", list(SLOPE_MAP.keys()))
        vessels = st.slider("Number of Major Vessels", 0, 3, 0)
        thallium_label = st.selectbox("Thallium Test", list(THAL_MAP.keys()))
        spo2 = st.slider("SpO₂", 80, 100, 96)
        temperature = st.slider("Temperature (°F)", 95.0, 105.0, 98.6, step=0.1)

    c4, c5 = st.columns(2)
    with c4:
        rr = st.slider("Respiratory Rate", 8, 35, 16)
    with c5:
        symptom_severity = st.slider("Symptom Severity Score", 1.0, 10.0, 5.0, step=0.1)
        activity_fatigue = st.slider("Activity / Fatigue Score", 1.0, 10.0, 5.0, step=0.1)

    sex = SEX_MAP[sex_label]
    chest_pain = CHEST_PAIN_MAP[chest_pain_label]
    fbs = FBS_MAP[fbs_label]
    ekg = EKG_MAP[ekg_label]
    exercise_angina = ANGINA_MAP[exercise_angina_label]
    slope = SLOPE_MAP[slope_label]
    thallium = THAL_MAP[thallium_label]

else:
    st.caption("Combined patient and clinician inputs for the most comprehensive assessment.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 20, 100, 50)
        sex_label = st.selectbox("Sex", list(SEX_MAP.keys()))
        spo2 = st.slider("SpO₂", 80, 100, 95)
        temperature = st.slider("Temperature (°F)", 95.0, 105.0, 99.0, step=0.1)
        rr = st.slider("Respiratory Rate", 8, 35, 18)

    with c2:
        symptom_severity = st.slider("Symptom Severity Score", 1.0, 10.0, 6.0, step=0.1)
        activity_fatigue = st.slider("Activity / Fatigue Score", 1.0, 10.0, 6.0, step=0.1)
        chest_pain_label = st.selectbox("Chest Pain Type", list(CHEST_PAIN_MAP.keys()))
        exercise_angina_label = st.selectbox("Exercise Angina", list(ANGINA_MAP.keys()))
        max_hr = st.slider("Max Heart Rate", 60, 220, 145)

    with c3:
        bp = st.slider("Resting Blood Pressure", 80, 220, 140)
        cholesterol = st.slider("Cholesterol", 100, 600, 250)
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(FBS_MAP.keys()))
        ekg_label = st.selectbox("EKG Results", list(EKG_MAP.keys()))
        st_depression = st.slider("ST Depression", 0.0, 6.5, 1.4, step=0.1)
        slope_label = st.selectbox("Slope of ST", list(SLOPE_MAP.keys()))
        vessels = st.slider("Number of Major Vessels", 0, 3, 1)
        thallium_label = st.selectbox("Thallium Test", list(THAL_MAP.keys()))

    sex = SEX_MAP[sex_label]
    chest_pain = CHEST_PAIN_MAP[chest_pain_label]
    fbs = FBS_MAP[fbs_label]
    ekg = EKG_MAP[ekg_label]
    exercise_angina = ANGINA_MAP[exercise_angina_label]
    slope = SLOPE_MAP[slope_label]
    thallium = THAL_MAP[thallium_label]

# =========================
# Build input frame
# =========================
input_dict = build_full_input_dict(
    age=age,
    sex=sex,
    chest_pain=chest_pain,
    bp=bp,
    cholesterol=cholesterol,
    fbs=fbs,
    ekg=ekg,
    max_hr=max_hr,
    exercise_angina=exercise_angina,
    st_depression=st_depression,
    slope=slope,
    vessels=vessels,
    thallium=thallium,
    spo2=spo2,
    temperature=temperature,
    rr=rr,
    symptom_severity=symptom_severity,
    activity_fatigue=activity_fatigue
)

input_df = pd.DataFrame([input_dict])
input_df = input_df[feature_names]

# =========================
# Prediction
# =========================
if st.button("Run AI Risk Assessment", use_container_width=True):
    scaled_input = scaler.transform(input_df)
    pred_prob = model.predict_proba(scaled_input)[0][1]

    final_risk_label, color = determine_risk(
        pred_prob=pred_prob,
        spo2=spo2,
        temperature=temperature,
        rr=rr,
        bp=bp,
        chest_pain=chest_pain,
        exercise_angina=exercise_angina
    )

    recommendation = recommendation_text(final_risk_label, mode)
    confidence_band = get_confidence_band(pred_prob)
    assessment_id = generate_assessment_id()
    escalation_status = get_escalation_status(final_risk_label)
    hospital_decision = get_hospital_recommendation(final_risk_label, spo2, temperature, rr, bp)

    add_to_history(assessment_id, mode, final_risk_label, pred_prob, hospital_decision["decision"])

    st.markdown("---")
    st.subheader("Assessment Results")

    show_alert_banner(final_risk_label)
    show_emergency_escalation(final_risk_label)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Predicted Risk", final_risk_label, color)
    with m2:
        metric_card("Model Probability", f"{pred_prob * 100:.1f}%", "#2D7FF9")
    with m3:
        metric_card("Confidence Band", confidence_band, "#7E57C2")
    with m4:
        metric_card("Assessment ID", assessment_id, "#607D8B")

    st.markdown("### Care Recommendation System")

    st.markdown(f"""
    <div style="
     background:#F4FAFF;
     padding:20px;
     border-radius:16px;
     border:1px solid #CFE8FF;
     box-shadow:0 4px 14px rgba(45,140,255,0.08);
     margin-bottom:16px;
    ">
     <h4 style="margin-top:0; color:#0B3C6D;">Recommended Next Step</h4>

     <p style="margin:6px 0; color:#1F2937;"><b>Disposition:</b> {hospital_decision["decision"]}</p>
     <p style="margin:6px 0; color:#1F2937;"><b>Clinical Recommendation:</b> {recommendation}</p>
     <p style="margin:6px 0; color:#1F2937;"><b>Escalation Status:</b> {escalation_status}</p>
     <p style="margin:6px 0; color:#1F2937;"><b>Reason:</b> {hospital_decision["reason"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Why This Prediction?")
    explanation_df = explain_prediction(input_df, model.feature_importances_)
    st.table(explanation_df)

    plain_drivers = get_plain_english_risk_drivers(explanation_df)
    drivers_html = "".join([f"<p style='margin:4px 0; color:#1F2937;'>• {item}</p>" for item in plain_drivers])
    section_box(
        "Primary Risk Contributors",
        drivers_html + "<p style='margin-top:10px; color:#374151;'><i>Risk elevated due to combined physiological abnormalities.</i></p>"
    )

    st.markdown("### Preventive Suggestions")
    suggestions = preventive_suggestions(final_risk_label)
    for item in suggestions:
        st.write(f"• {item}")

    st.markdown("### User-Friendly Monitoring Summary")
    summary_df = user_friendly_summary(
    mode=mode,
    age=age,
    sex_label=sex_label,
    spo2=spo2,
    temperature=temperature,
    rr=rr,
    symptom_severity=symptom_severity,
    activity_fatigue=activity_fatigue
    )
    st.table(summary_df)

    render_shareable_summary(
        assessment_id=assessment_id,
        risk_label=final_risk_label,
        spo2=spo2,
        temperature=temperature,
        rr=rr,
        recommendation=hospital_decision["decision"]
    )

    st.markdown("### Simulated Monitoring Trend")

    trend_df = pd.DataFrame({
     "Time": ["T-3", "T-2", "T-1", "Current"],
     "SpO2": [min(100, spo2 + 2), min(100, spo2 + 1), spo2, spo2],
     "Temperature_F": [max(95.0, temperature - 0.6), max(95.0, temperature - 0.3), temperature - 0.1, temperature],
     "Respiratory_Rate": [max(8, rr - 3), max(8, rr - 2), max(8, rr - 1), rr]
    })

    col1, col2, col3 = st.columns(3)

    with col1:
     fig1 = px.line(trend_df, x="Time", y="SpO2", markers=True, title="SpO₂ Trend")
     fig1.update_layout(
        plot_bgcolor="#EAF6FF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0B3C6D"),
        margin=dict(l=20, r=20, t=40, b=20)
     )
     st.plotly_chart(fig1, use_container_width=True)

    with col2:
     fig2 = px.line(trend_df, x="Time", y="Temperature_F", markers=True, title="Temperature Trend")
     fig2.update_layout(
        plot_bgcolor="#EAF6FF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0B3C6D"),
        margin=dict(l=20, r=20, t=40, b=20)
     )
     st.plotly_chart(fig2, use_container_width=True)

    with col3:
     fig3 = px.line(trend_df, x="Time", y="Respiratory_Rate", markers=True, title="Respiratory Rate Trend")
     fig3.update_layout(
        plot_bgcolor="#EAF6FF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0B3C6D"),
        margin=dict(l=20, r=20, t=40, b=20)
     )
     st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Assessments")
    history_df = pd.DataFrame(st.session_state.assessment_history)
    st.table(history_df)

    st.markdown("### Clinician Notes")

    default_observation = ""
    default_advice = ""

    if final_risk_label == "LOW RISK":
        default_observation = "Patient stable with no immediate red-flag indicators."
        default_advice = "Continue home monitoring and repeat assessment if symptoms worsen."
    elif final_risk_label == "MODERATE RISK":
        default_observation = "Moderate clinical concern based on current physiological and symptom indicators."
        default_advice = "Recommend clinician consultation and closer monitoring over the next few hours."
    else:
        default_observation = "High-risk presentation with escalation indicators."
        default_advice = "Immediate physician review and urgent care referral recommended."

    st.text_area("Clinician Observation", value=default_observation, height=120)
    st.text_area("Follow-up Advice", value=default_advice, height=120)

else:
    st.info("Select a mode, enter patient details, and click **Run AI Risk Assessment**.")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Design Note: This prototype uses role-appropriate interfaces. "
    "Common users interact with understandable wearable and symptom-based inputs, "
    "while advanced clinical parameters are reserved for healthcare professionals. "
    "This system supports, not replaces, clinical judgment."
)