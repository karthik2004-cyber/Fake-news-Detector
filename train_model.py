import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
dataset_path = "fake_and_real_news.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: Dataset file '{dataset_path}' not found!")

df = pd.read_csv(dataset_path)
print("Dataset loaded successfully!")
print("Columns in the dataset:", df.columns)  # Print the column names

# Ensure correct column names
if "label" not in df.columns:
    if "category" in df.columns:
        df.rename(columns={"category": "label"}, inplace=True)
    else:
        raise ValueError("Error: No 'label' or 'category' column found!")

if "text" not in df.columns:
    if "content" in df.columns:
        df.rename(columns={"content": "text"}, inplace=True)
    else:
        raise ValueError("Error: No 'text' or 'content' column found!")

# Filter dataset
df = df[df["label"].str.upper().isin(["FAKE", "REAL"])]
df.dropna(subset=["text"], inplace=True)
df["label"] = df["label"].str.upper().map({"FAKE": 0, "REAL": 1})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save Model & Vectorizer
try:
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")

try:
    with open("vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print("Vectorizer saved successfully!")
except Exception as e:
    print(f"Error saving vectorizer: {e}")

# Verify Saved Files
print("\nChecking saved files:")
print("Model file exists:", os.path.exists("model.pkl"))
print("Vectorizer file exists:", os.path.exists("vectorizer.pkl"))
