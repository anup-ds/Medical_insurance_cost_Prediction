import streamlit as st
import joblib
import numpy as np
import pandas as pd



st.title("Medical Insurance Cost Prediction")

st.image("https://media.istockphoto.com/id/1226082621/photo/insurance-concept-stack-of-wooden-blocks-with-words-life-health-legal-expenses-business-house.jpg?s=612x612&w=0&k=20&c=5bKk7pRl9jewZM_nmIquyGOj4Q7BVNiYRcJC9H1smfE="
         , caption="Medical Insurance",width=250)

# Input features

age = st.slider("Age", 18, 70, 30)
sex = st.radio("Sex", ["Male", "Female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# load the model and scaler
model = joblib.load('life_insurance_model.pkl')
scaler = joblib.load('scaler.pkl')

# prediction button

    
# To run the app, use the command: streamlit run app1.py
if st.button("Predict Insurance Cost"):
    # Preprocess input features for prediction
    encoded_sex = 1 if sex == "Male" else 0
    encoded_smoker = 0 if smoker == "Yes" else 1
    # One-hot encoding for region
    region_list = ["Northeast", "Northwest", "Southeast", "Southwest"]
    encoded_region = [1 if region == r else 0 for r in region_list]
    # Combine all features
    insurance_input = [[age, encoded_sex, bmi, children, encoded_smoker] + encoded_region]
    scaled = scaler.transform(insurance_input)
    prediction = model.predict(scaled)

    # Inverse transform the prediction to get original scale
    predicted_original_cost = np.expm1(prediction)
    st.success(f"The predicted insurance cost is: {predicted_original_cost[0]:.2f} Rs")

