from flask import Flask, request, jsonify 
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load trained model & vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    app.logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model or vectorizer: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["text"]
        transformed_text = vectorizer.transform([data])
        prediction = model.predict(transformed_text)[0]
        result = "Real" if prediction == 1 else "Fake"
        app.logger.info(f"Prediction made successfully: {result}")
        return jsonify({"prediction": result})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "UP"}), 200

if __name__ == "__main__":
    try:
        app.logger.info("Starting the server...")
        app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        app.logger.error(f"Error starting the server: {e}")


