import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_project_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for "Visit with Us" staff that predicts whether predicting the package for customers based on their details.")
st.write("Kindly enter the customer details to check whether they are likely purchase package.")

# Collect user input
CreditScore = st.number_input("Credit Score (customer's credit score)", min_value=300, max_value=900, value=650)
Geography = st.selectbox("Geography (country where the customer resides)", ["France", "Germany", "Spain"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (number of years the customer has been with the bank)", value=12)
Balance = st.number_input("Account Balance (customer\u2019s account balance)", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products (number of products the customer has with the bank)", min_value=1, value=1)
HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])
IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"])
EstimatedSalary = st.number_input("Estimated Salary (customer\u2019s estimated salary)", min_value=0.0, value=50000.0)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact":TypeofContact
    "CityTier":CityTier
    "DurationOfPitch":DurationOfPitch
    "Occupation":Occupation
    "Gender":Gender
    "NumberOfPersonVisiting":NumberOfPersonVisiting
    "NumberOfFollowups":NumberOfFollowups
    "ProductPitched":ProductPitched
    "PreferredPropertyStar":PreferredPropertyStar
    "MaritalStatus": "Married" if MaritalStatus == "Married" else "Single",
    "NumberOfTrips":NumberOfTrips
    "Passport":1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore":PitchSatisfactionScore
    "OwnCar":1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting":NumberOfChildrenVisiting
    "Designation":Designation
    "MonthlyIncome":MonthlyIncome
}])


# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Not Buy Package" if prediction == 1 else "Buy Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
