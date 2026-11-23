import os
import streamlit as st
import pandas as pd
import joblib

st.title("Tourism Package Purchase Prediction")
st.write("Enter customer details to predict if they will purchase the newly introduced Wellness Tourism Package.")

# Model Path
model_path = "tourism_project/deployment/tourism_project_model_v1.joblib"

if not os.path.exists(model_path):
    st.warning("Model file not found. Please ensure the pipeline has run and committed the model v1 file.")
else:
    model = joblib.load(model_path)

    # User Inputs corresponding to the training dataset features
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=150, value=15)
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 3)
    number_of_followups = st.slider("Number of Follow-ups", 1, 6, 4)
    preferred_property_star = st.selectbox("Preferred Property Hotel Star Rating", [3.0, 4.0, 5.0], index=0)
    number_of_trips = st.number_input("Average Annual Trips", min_value=1, max_value=30, value=3)
    
    # Convert Yes/No string selections back to 1/0 for modeling binary flags
    passport_str = st.selectbox("Holds Valid Passport?", ["Yes", "No"], index=1)
    passport = 1 if passport_str == "Yes" else 0

    pitch_satisfaction_score = st.selectbox("Sales Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
    
    own_car_str = st.selectbox("Owns a Car?", ["Yes", "No"], index=0)
    own_car = 1 if own_car_str == "Yes" else 0

    number_of_children_visiting = st.slider("Number of Children (under age 5) Visiting", 0, 3, 1)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=22000)

    # Categorical Inputs
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

    # Gather inputs into DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "PreferredPropertyStar": [preferred_property_star],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "MonthlyIncome": [monthly_income],
        "TypeofContact": [typeof_contact],
        "Occupation": [occupation],
        "Gender": [gender],
        "ProductPitched": [product_pitched],
        "MaritalStatus": [marital_status],
        "Designation": [designation]
    })

    if st.button("Predict Purchase Likelihood"):
        prediction_proba = model.predict_proba(input_data)[0][1]
        classification_threshold = 0.45
        prediction = 1 if prediction_proba >= classification_threshold else 0
        
        st.subheader("Prediction Results")
        if prediction == 1:
            st.success(f"Likely to Purchase (Probability: {prediction_proba:.2%})")
        else:
            st.info(f"Not Likely to Purchase (Probability: {prediction_proba:.2%})")
