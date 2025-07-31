# src/genre_classifier.py

import os
import re
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess dataset
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _, title, genre, description = parts
                full_text = f"{title} {description}"
                texts.append(full_text)
                labels.append(genre)
    return texts, labels

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()

# Paths
data_path = "data/Genre Classification Dataset/train_data.txt"
model_output_path = "models/genre_classifier.pkl"

# Load and clean data
texts, labels = load_data(data_path)
texts = [clean_text(t) for t in texts]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split data
X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer with bigrams
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = accuracy_score(y_train, model.predict(X_train)) * 100
test_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

# Save model
with open(model_output_path, "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }, f)

# Print results
print(f"✅ Train Accuracy: {train_accuracy:.2f}%")
print(f"✅ Test Accuracy: {test_accuracy:.2f}%")
print("✅ Model trained and saved to:", model_output_path)
