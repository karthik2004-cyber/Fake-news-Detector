
from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    transformed_text = vectorizer.transform([data])
    prediction = model.predict(transformed_text)[0]
    result = "Real" if prediction == 1 else "Fake"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


