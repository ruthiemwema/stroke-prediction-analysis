import streamlit as st
import pandas as pd
import joblib

model = joblib.load('stroke_model.joblib')
scaler = joblib.load('scaler.joblib')
columns = joblib.load('feature_columns.joblib')

THRESHOLD = 0.6

st.title("Stroke Risk Screening Demo")
st.write("Enter patient details to get a risk score from a logistic regression model.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    glucose = st.number_input("Average glucose level", min_value=40.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    married = st.selectbox("Ever married", ["Yes", "No"])

with col2:
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
    work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence type", ["Urban", "Rural"])
    smoking = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

if st.button("Calculate risk"):
    row = dict.fromkeys(columns, 0)
    row['age'] = age
    row['avg_glucose_level'] = glucose
    row['bmi'] = bmi
    row['hypertension'] = 1 if hypertension == "Yes" else 0
    row['heart_disease'] = 1 if heart_disease == "Yes" else 0
    row['ever_married_Yes'] = 1 if married == "Yes" else 0
    row['gender_Male'] = 1 if gender == "Male" else 0
    row['gender_Other'] = 1 if gender == "Other" else 0
    row['Residence_type_Urban'] = 1 if residence == "Urban" else 0

    # Govt_job and Unknown smoking are the baseline categories (all zeros)
    for key in [f"work_type_{work_type}", f"smoking_status_{smoking}"]:
        if key in row:
            row[key] = 1

    X_input = pd.DataFrame([row])[columns]
    score = model.predict_proba(scaler.transform(X_input))[0, 1]

    st.metric("Risk score", f"{score:.0%}")
    if score >= THRESHOLD:
        st.error("Flagged as HIGH risk. Follow-up screening is recommended.")
    else:
        st.success("Not flagged as high risk.")

st.caption(
    "Demo only, not medical advice. The score comes from a model trained on SMOTE-balanced "
    "data, so it works as a relative risk ranking, not a true probability."
)