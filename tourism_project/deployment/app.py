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
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("The method by which the customer was contacted", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("The city category based on development, population, and living standards.", ["1", "2", "3"])
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer.", min_value=5, max_value=33, value=5)
Occupation = st.selectbox("Customer's occupation.", ["Small Business", "Large Business", "Salaried"])
Gender = st.selectbox("Gender of the customer.", ["Male", "Female"])
NumberOfPersonVisiting = st.selectbox("Total number of people accompanying the customer on the trip.", ["1", "2", "3", "4"])
NumberOfFollowups = st.selectbox("Total number of follow-ups by the salesperson after the sales pitch.", ["1", "2", "3","4", "5", "6"])
ProductPitched = st.selectbox("The type of product pitched to the customer.", ["King", "Basic", "Deluxe", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer.", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital status of the customer.", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Duration of the sales pitch delivered to the customer.", min_value=0, max_value=50, value=1)
Passport = st.selectbox("Whether the customer holds a valid passport.", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Score indicating the customer's satisfaction with the sales pitch.", ["Yes", "No"])
OwnCar = st.selectbox("Whether the customer owns a car.", ["Yes", "No"])
NumberOfChildrenVisiting = st.selectbox("Geography (country where the customer resides)", ["0", "1", "2", "3"])
Designation = st.selectbox("Geography (country where the customer resides)", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Estimated monthly Income:", min_value=0.0, value=50000.0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact":TypeofContact,
    "CityTier":CityTier,
    "DurationOfPitch":DurationOfPitch,
    "Occupation":Occupation,
    "Gender":Gender,
    "NumberOfPersonVisiting":NumberOfPersonVisiting,
    "NumberOfFollowups":NumberOfFollowups,
    "ProductPitched":ProductPitched,
    "PreferredPropertyStar":PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips":NumberOfTrips,
    "Passport":1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore":PitchSatisfactionScore,
    "OwnCar":1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting":NumberOfChildrenVisiting,
    "Designation":Designation,
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
