"""
EduPredict — Exam Score Prediction
Streamlit UI for Feature Story 5.

Expects two files produced during training (Feature Story 5, Task 1):
    - model_pipeline.pkl   -> joblib-dumped sklearn Pipeline (StandardScaler + best model)
    - expected_columns.pkl -> joblib-dumped list[str], the exact column order the
                               pipeline was trained on (after encoding)

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="centered")

MODEL_PATH = "model/model_pipeline.pkl"
COLUMNS_PATH = "model/expected_columns.pkl"
AT_RISK_THRESHOLD = 60  # tweak to match your institution's passing score

ORDINAL_MAPS = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
}

NOMINAL_COLUMNS = [
    "Gender",
    "School_Type",
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Peer_Influence",
]

NOMINAL_OPTIONS = {
    "Gender": ["Male", "Female"],
    "School_Type": ["Public", "Private"],
    "Extracurricular_Activities": ["Yes", "No"],
    "Internet_Access": ["Yes", "No"],
    "Learning_Disabilities": ["Yes", "No"],
    "Peer_Influence": ["Positive", "Neutral", "Negative"],
}

NUMERIC_RANGES = {
    "Hours_Studied": (0, 40, 20),
    "Attendance": (0, 100, 85),
    "Sleep_Hours": (0, 12, 7),
    "Previous_Scores": (0, 100, 70),
    "Tutoring_Sessions": (0, 10, 1),
    "Physical_Activity": (0, 10, 3),
}


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained pipeline and expected column order. Returns (None, None, error) on failure."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        expected_columns = joblib.load(COLUMNS_PATH)
        return pipeline, expected_columns, None
    except FileNotFoundError as e:
        return None, None, f"Missing file: {e.filename}"
    except Exception as e:  # pragma: no cover
        return None, None, str(e)


def build_feature_row(raw_inputs: dict, expected_columns: list[str]) -> pd.DataFrame:
    """Apply the same ordinal + one-hot encoding used at training time and
    align to the exact column order the pipeline expects."""
    row = {}

    # Numeric passthrough
    for col in NUMERIC_RANGES:
        row[col] = raw_inputs[col]

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        row[col] = mapping[raw_inputs[col]]

    # One-hot encoding (drop_first not assumed — we set every dummy column to 0,
    # then flip the one that matches the selected value)
    for col in NOMINAL_COLUMNS:
        selected = raw_inputs[col]
        for option in NOMINAL_OPTIONS[col]:
            dummy_name = f"{col}_{option}"
            row[dummy_name] = 1 if option == selected else 0

    df = pd.DataFrame([row])

    # Align to training columns: add any missing (e.g. a dropped dummy level) as 0,
    # drop anything extra, and enforce the exact training order.
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    return df


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🎓 EduPredict")
st.caption("Estimate a student's exam score from study habits, resources, and context.")

pipeline, expected_columns, load_error = load_artifacts()

if load_error:
    st.error(
        f"Couldn't load the trained model ({load_error}). "
        f"Place `{MODEL_PATH}` and `{COLUMNS_PATH}` (from `joblib.dump()` in "
        f"Feature Story 5) next to `app.py` and reload."
    )
    st.stop()

with st.form("student_profile"):
    st.subheader("Study habits")
    c1, c2 = st.columns(2)
    with c1:
        hours_studied = st.slider("Hours studied / week", *NUMERIC_RANGES["Hours_Studied"])
        attendance = st.slider("Attendance (%)", *NUMERIC_RANGES["Attendance"])
        sleep_hours = st.slider("Sleep hours / night", *NUMERIC_RANGES["Sleep_Hours"])
    with c2:
        previous_scores = st.slider("Previous scores", *NUMERIC_RANGES["Previous_Scores"])
        tutoring_sessions = st.slider("Tutoring sessions / month", *NUMERIC_RANGES["Tutoring_Sessions"])
        physical_activity = st.slider("Physical activity (hrs/week)", *NUMERIC_RANGES["Physical_Activity"])

    st.subheader("Resources & environment")
    c3, c4 = st.columns(2)
    with c3:
        parental_involvement = st.selectbox("Parental involvement", list(ORDINAL_MAPS["Parental_Involvement"]))
        access_to_resources = st.selectbox("Access to resources", list(ORDINAL_MAPS["Access_to_Resources"]))
        family_income = st.selectbox("Family income", list(ORDINAL_MAPS["Family_Income"]))
        teacher_quality = st.selectbox("Teacher quality", list(ORDINAL_MAPS["Teacher_Quality"]))
    with c4:
        motivation_level = st.selectbox("Motivation level", list(ORDINAL_MAPS["Motivation_Level"]))
        parental_education = st.selectbox("Parental education level", list(ORDINAL_MAPS["Parental_Education_Level"]))
        distance_from_home = st.selectbox("Distance from home", list(ORDINAL_MAPS["Distance_from_Home"]))
        internet_access = st.selectbox("Internet access", NOMINAL_OPTIONS["Internet_Access"])

    st.subheader("Background")
    c5, c6 = st.columns(2)
    with c5:
        gender = st.selectbox("Gender", NOMINAL_OPTIONS["Gender"])
        school_type = st.selectbox("School type", NOMINAL_OPTIONS["School_Type"])
    with c6:
        extracurricular = st.selectbox("Extracurricular activities", NOMINAL_OPTIONS["Extracurricular_Activities"])
        learning_disabilities = st.selectbox("Learning disabilities", NOMINAL_OPTIONS["Learning_Disabilities"])
        peer_influence = st.selectbox("Peer influence", NOMINAL_OPTIONS["Peer_Influence"])

    submitted = st.form_submit_button("Predict exam score", use_container_width=True)

if submitted:
    raw_inputs = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "Motivation_Level": motivation_level,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
        "Internet_Access": internet_access,
        "Gender": gender,
        "School_Type": school_type,
        "Extracurricular_Activities": extracurricular,
        "Learning_Disabilities": learning_disabilities,
        "Peer_Influence": peer_influence,
    }

    try:
        features = build_feature_row(raw_inputs, expected_columns)
        prediction = float(pipeline.predict(features)[0])
        prediction = float(np.clip(prediction, 0, 100))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Predicted exam score", f"{prediction:.1f} / 100")

    if prediction < AT_RISK_THRESHOLD:
        m2.error("⚠️ At risk of failing")
        st.warning(
            f"Predicted score is below the {AT_RISK_THRESHOLD} threshold. "
            "Consider flagging this student for additional support."
        )
    else:
        m2.success("✅ On track")
