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

        result_label.config(text=f"Prediction: {result}", fg="#00ff00" if prediction == 1 else "#ff4d4d")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize GUI
root = tk.Tk()
root.title("Fake News Detection System")
root.geometry("600x500")
root.configure(bg="#1e1e1e")

# Styling
style_bg = "#2a2a2a"
style_fg = "#ffffff"
style_btn_bg = "#00d9ff"
style_btn_hover = "#007bff"

frame = tk.Frame(root, bg=style_bg, padx=20, pady=20)
frame.pack(pady=20)

title_label = tk.Label(frame, text="Fake News Detector", font=("Arial", 20, "bold"), fg=style_fg, bg=style_bg)
title_label.pack(pady=10)

text_input = scrolledtext.ScrolledText(frame, height=8, width=60, font=("Arial", 12), bg="#333333", fg=style_fg, insertbackground=style_fg)
text_input.pack(pady=10, padx=10)

def on_enter(e):
    predict_btn.config(bg=style_btn_hover)

def on_leave(e):
    predict_btn.config(bg=style_btn_bg)

predict_btn = tk.Button(frame, text="Check News", font=("Arial", 14, "bold"), bg=style_btn_bg, fg=style_fg, padx=15, pady=7, command=predict_news, relief="flat", bd=3)
predict_btn.pack(pady=10)
predict_btn.bind("<Enter>", on_enter)
predict_btn.bind("<Leave>", on_leave)

result_label = tk.Label(frame, text="Prediction: ", font=("Arial", 14, "bold"), fg=style_fg, bg=style_bg)
result_label.pack(pady=10)

root.mainloop()
