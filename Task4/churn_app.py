import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn (0 = No Churn, 1 = Churn)")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure (years with bank)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_crcard = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0)

# When Predict button is pressed
if st.button("Predict Churn"):
    # Create dataframe with correct column names
    input_dict = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [has_crcard],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [salary]
    }
    input_df = pd.DataFrame(input_dict)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Show results
    st.subheader(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {probability:.2%}")
