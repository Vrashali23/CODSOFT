# spam_app.py
import streamlit as st
import pickle

# Load model and vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("spam_classifier_svm.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="📩 Spam SMS Detector", layout="centered")

st.title("📩 Spam SMS Detector")
st.write("Enter your SMS below and check whether it's spam or not.")

user_input = st.text_area("✉️ SMS Message", height=150)

if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚫 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM (Legitimate)**.")
