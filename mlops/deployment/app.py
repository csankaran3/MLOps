import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="csankaran3/engine_condition_prediction", filename="best_engine_condition_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("Predictive Maintenance - Engine fault prediction application")
st.write("This App is an internal tool for automobie companies to predict engine condition (Active / Faulty) based on the sensor values.")
st.subheader("Kindly enter the sensor details to check whether engine condition is active or faulty.")

engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=750.0, step=1.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure (kPa)", min_value=0.0, value=3.10, step=0.01)
fuel_pressure = st.number_input("Fuel Pressure (kPa)", min_value=0.0, value=6.20, step=0.01)
coolant_pressure = st.number_input("Coolant Pressure (kPa)", min_value=0.0, value=2.10, step=0.01)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, value=76.80, step=0.01)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, value=78.30, step=0.01)


# Convert inputs to match model training
input_data = pd.DataFrame([{
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp
}])


# Predict button
if st.button("Predict Engine Condition"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Active" if prediction == 1 else "Faulty"
    st.success(f"Engine condition prediction completed!.. The Engine condition is {result}.")
