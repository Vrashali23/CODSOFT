import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
st.success("✅ Model loaded successfully!")

# Input fields for all 11 features
st.subheader("🔍 Manual Input")

amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
gender = st.selectbox("Gender", [0, 1])  # 0: F, 1: M
category = st.number_input("Category (numeric)", min_value=0, value=2)
zip_code = st.number_input("ZIP Code", min_value=10000, value=94102)
latitude = st.number_input("Latitude", value=37.77)
longitude = st.number_input("Longitude", value=-122.42)
unix_time = st.number_input("Unix Time", value=1325376000)
merchant_category = st.number_input("Merchant Category (encoded)", value=3)
state_encoded = st.number_input("State (encoded)", value=5)
city_encoded = st.number_input("City (encoded)", value=10)
hour = st.slider("Transaction Hour", 0, 23, 12)

# Predict button
if st.button("Predict Fraud"):
    input_data = np.array([[amt, gender, category, zip_code, latitude, longitude,
                            unix_time, merchant_category, state_encoded, city_encoded, hour]])
    
    prediction = model.predict(input_data)[0]
    
    result = "🔴 Fraudulent Transaction" if prediction == 1 else "🟢 Legitimate Transaction"
    
    st.subheader("Prediction Result")
    st.info(result)
