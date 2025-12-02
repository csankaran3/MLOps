import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="csankaran3/churn-model", filename="best_churn_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for 'Visit with us' travel company staff that predicts potential package buyers.")
st.write("Kindly enter the customer details to check whether they are likely to buy the package or not.")

# List of numerical features in the dataset
numeric_features = [
    'Age',               # Customer's age
    'CityTier',          # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
    'DurationOfPitch',   # Duration of the sales pitch delivered to the customer.
    'NumberOfPersonVisiting', # Total number of people accompanying the customer on the trip.
    'NumberOfFollowups',      # Total number of follow-ups by the salesperson after the sales pitch.
    'PreferredPropertyStar',  # Preferred hotel rating by the customer.
    'NumberOfTrips',          # Average number of trips the customer takes annually.
    'Passport',                # Whether the customer holds a valid passport (0: No, 1: Yes).
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar',                 # Whether the customer owns a car (0: No, 1: Yes).
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer.
    'MonthlyIncome'             # Gross monthly income of the customer.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',      # Method by which the customer was contacted (Company Invited or Self Inquiry).
    'Occupation',         # Customer's occupation (e.g., Salaried, Freelancer).
    'Gender',             # Gender of the customer (Male, Female).
    'ProductPitched',     # The type of product pitched to the customer.
    'MaritalStatus',      # Marital status of the customer (Single, Married, Divorced).
    'Designation'         # Customer's designation in their current organization.
]

st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
        Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
        CityTier = st.selectbox("City Tier", options=["Tier 1", "Tier 2", "Tier 3"], index=1)
        NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=1)
        PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
        NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=10, value=1)
        Passport = st.selectbox("Holding Passport?", options=["Yes", "No"], index=0)
        OwnCar = st.selectbox("Owns a Car?", options=["Yes", "No"], index=0)
    
with col2:
        NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
        MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
        TypeofContact = st.selectbox("TypeofContact", options=["Company Invited", "Self Enquiry"], index=0)
        Occupation = st.selectbox("Occupation", options=["Salaried", "Small Business", "Large Business", "Freelancer"], index=0)
        Gender = st.selectbox("Gender", options=["Male", "Female"], index=1)
        MaritalStatus = st.selectbox("MaritalStatus", options=["Single", "Married", "Unmarried","Divorced"], index=1)
        Designation = st.selectbox("Designation", options=["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=0)

st.subheader("Customer Interaction Data")
col3, col4 = st.columns(2)

with col3:
        PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=0)

with col4:
        ProductPitched = st.selectbox("ProductPitched", options=["Basic", "Deluxe", "Premium", "Super Deluxe", "King"], index=1)
        DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=10, value=3)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
        'Age': Age,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'NumberofPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'PreferredPropertyStar': PreferredPropertyStar,
        'NumberOfTrips': NumberOfTrips,
        'Passport': 1 if Passport == "Yes" else 0,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': 1 if OwnCar == "Yes" else 0,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'MonthlyIncome': MonthlyIncome,
        'TypeofContact': TypeofContact,
        'Occupation': Occupation,
        'Gender': Gender,
        'ProductPitched': ProductPitched,
        'MaritalStatus': MaritalStatus,
        'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
#if st.button("Predict"):
#    prediction_proba = model.predict_proba(input_data)[0, 1]
#    prediction = (prediction_proba >= classification_threshold).astype(int)
#    result = "Likely to purchase the package" if prediction == 1 else " Unlikely to purchase the package"
#    st.write(f"Based on the information provided, the customer is {result}.")

# Predict button
if st.button("Predict"):
    input_data.columns
    prediction = model.predict(input_data)[0]
    result = "Likely to purchase the package" if prediction == 1 else " Unlikely to purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"Based on the information provided, the customer is {result}.")
