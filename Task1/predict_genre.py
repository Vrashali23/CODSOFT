# predict_genre.py

import pickle

# Load trained model
with open("models/genre_classifier.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
vectorizer = data["vectorizer"]
label_encoder = data["label_encoder"]

# Enter movie title & description
title = input("Enter movie title: ")
description = input("Enter movie description: ")
full_text = f"{title} {description}"

# Vectorize and predict
X = vectorizer.transform([full_text])
pred = model.predict(X)
predicted_genre = label_encoder.inverse_transform(pred)[0]

print("🎬 Predicted Genre:", predicted_genre)
