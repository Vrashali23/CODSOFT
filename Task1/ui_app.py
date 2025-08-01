# ui_app.py

import streamlit as st
import pickle

# Load model
with open("models/genre_classifier.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
vectorizer = data["vectorizer"]
label_encoder = data["label_encoder"]

# UI
st.title("🎬 IMDb Genre Classifier")

title = st.text_input("Enter Movie Title")
desc = st.text_area("Enter Movie Description")

if st.button("Predict Genre"):
    full_text = f"{title} {desc}"
    X = vectorizer.transform([full_text])
    pred = model.predict(X)
    genre = label_encoder.inverse_transform(pred)[0]
    st.success(f"Predicted Genre: {genre}")
