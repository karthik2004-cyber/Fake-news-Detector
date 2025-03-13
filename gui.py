import tkinter as tk
from tkinter import messagebox, scrolledtext
import pickle
import os
import sys

# Check if model and vectorizer exist
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    messagebox.showerror("Error", "Model or Vectorizer file not found! Ensure 'model.pkl' and 'vectorizer.pkl' exist.")
    sys.exit()

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model/vectorizer: {e}")
    sys.exit()

def predict_news():
    news_text = text_input.get("1.0", tk.END).strip()
    if not news_text:
        messagebox.showwarning("Input Error", "Please enter some news text!")
        return

    try:
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        result = "Real News" if prediction == 1 else "Fake News"

        result_label.config(text=f"Prediction: {result}", fg="green" if prediction == 1 else "red")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize GUI
root = tk.Tk()
root.title("Fake News Detection System")
root.geometry("550x450")
root.configure(bg="#e6e6e6")

title_label = tk.Label(root, text="Fake News Detection System", font=("Arial", 18, "bold"), bg="#e6e6e6")
title_label.pack(pady=10)

text_input = scrolledtext.ScrolledText(root, height=8, width=60, font=("Arial", 12))
text_input.pack(pady=10, padx=10)

predict_btn = tk.Button(root, text="Check News", font=("Arial", 14, "bold"), bg="#007BFF", fg="white", padx=10, pady=5, command=predict_news)
predict_btn.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14, "bold"), bg="#e6e6e6")
result_label.pack(pady=10)

root.mainloop()
