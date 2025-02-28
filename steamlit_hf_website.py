import streamlit as st
import pandas as pd
import pickle

# Load trained CatBoost model
@st.cache_resource
def load_model():
    with open("/Users/yuthishkumar/Downloads/catboost_heart_failure.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App interface
st.title("Heart Failure Prediction")

# Collect user input
age = st.slider("Age", 20, 100, 50)
anaemia = st.radio("Anaemia", ["No", "Yes"])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=8000, value=200)
diabetes = st.radio("Diabetes", ["No", "Yes"])
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 40)
high_blood_pressure = st.radio("High Blood Pressure", ["No", "Yes"])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=50000, max_value=500000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=110, max_value=150, value=137)
sex = st.radio("Sex", ["Male", "Female"])
smoking = st.radio("Smoking", ["No", "Yes"])

# Convert categorical inputs to numerical values
input_data = pd.DataFrame({
    "age": [age],
    "anaemia": [1 if anaemia == "Yes" else 0],
    "creatinine_phosphokinase": [creatinine_phosphokinase],
    "diabetes": [1 if diabetes == "Yes" else 0],
    "ejection_fraction": [ejection_fraction],
    "high_blood_pressure": [1 if high_blood_pressure == "Yes" else 0],
    "platelets": [platelets],
    "serum_creatinine": [serum_creatinine],
    "serum_sodium": [serum_sodium],
    "sex": [1 if sex == "Male" else 0],
    "smoking": [1 if smoking == "Yes" else 0],
})

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "High Risk of Heart Failure" if prediction[0] == 1 else "Low Risk of Heart Failure"
    st.subheader(f"Prediction: {result}")
