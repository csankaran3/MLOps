import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="csankaran3/engine-condition-prediction", filename="best_engine_condition_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("Predictive Maintenance - Engine fault prediction application")
st.write("This App is an internal tool for automobie companies to predict engine condition (Active / Faulty) based on the sensor values.")
st.subheader("Kindly enter the sensor details to check whether engine condition is active or faulty.")

# Setting the minimum value and distplay value - Used min and average from the dataset for displaying values
engine_rpm = st.number_input("Engine RPM", min_value=61.0, value=1150.0, step=10.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure (kPa)", min_value=0.0, value=3.63, step=0.01)
fuel_pressure = st.number_input("Fuel Pressure (kPa)", min_value=0.0, value=10.57, step=0.01)
coolant_pressure = st.number_input("Coolant Pressure (kPa)", min_value=0.0, value=7.48, step=0.01)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=71.32, value=89.58, step=0.01)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=61.67, value=128.60, step=0.01)


# Convert inputs to match model training
input_data = pd.DataFrame([{
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict Engine Condition"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Active" if prediction == 1 else "Faulty"
    if (result == "Active"):
        st.success(f"Engine condition prediction completed!.. The Engine condition is {result}.")
    else:
        st.error(f"Engine condition prediction completed!.. The Engine condition is {result}.")

