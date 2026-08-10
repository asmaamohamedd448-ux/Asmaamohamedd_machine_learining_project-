import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

@st.cache_resource
def load_artifact():
    return joblib.load("model.pkl")

artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
gender_map = artifact["gender_map"]
smoking_map = artifact["smoking_map"]
feature_cols = artifact["feature_cols"]
scale_cols = artifact["scale_cols"]

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter the values below and click **Predict** to estimate diabetes risk. "
    "This is a machine-learning demo trained on a public dataset — "
    "**not a medical diagnosis.**"
)

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", list(gender_map.keys()))
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)

    with col2:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
        smoking = st.selectbox("Smoking history", list(smoking_map.keys()))
        glucose = st.number_input("Blood glucose level", min_value=50, max_value=400, value=100)

    threshold = st.slider(
        "Decision threshold (lower = flags more people as at-risk, catches more true cases but more false alarms)",
        min_value=0.05, max_value=0.95, value=0.5, step=0.05,
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "gender": gender_map[gender],
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_map[smoking],
        "bmi": bmi,
        "hbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }])[feature_cols]

    row[scale_cols] = scaler.transform(row[scale_cols])

    proba = model.predict_proba(row)[0, 1]
    prediction = int(proba >= threshold)

    st.divider()
    st.metric("Estimated diabetes risk probability", f"{proba:.1%}")

    if prediction == 1:
        st.error("⚠️ Model flags this as **higher risk** for diabetes.")
    else:
        st.success("✅ Model flags this as **lower risk** for diabetes.")

    st.caption(
        "This tool is trained on a public dataset for demonstration purposes only "
        "and is not a substitute for professional medical advice."
    )
