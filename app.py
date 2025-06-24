from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# 🚀 ML (Traditional) model + scaler
ml_model = joblib.load("diabetes_model.pkl")
ml_scaler = joblib.load("scaler.pkl")

# 🦁 DL (Lion) model + scaler
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

dl_model = DiabetesModel()
dl_model.load_state_dict(torch.load("diabetes_model_lion_upgraded.pth", map_location=torch.device("cpu")))
dl_model.eval()
dl_scaler = joblib.load("scaler_lion_upgraded.pkl")

# 📌 Shared fields
fields = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# 🧠 Route 1 - Traditional ML model
@app.route("/predict/ml", methods=["POST"])
def predict_ml():
    try:
        data = request.get_json()
        values = [[data[field] for field in fields]]
        df = pd.DataFrame(values, columns=fields)
        scaled = ml_scaler.transform(df)
        prediction = ml_model.predict(scaled)[0]
        return jsonify({
            "model": "traditional_ML",
            "result": "Likely Diabetic" if prediction == 1 else "Not Diabetic",
            "raw": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🦁 Route 2 - Deep Learning Lion model
@app.route("/predict/dl", methods=["POST"])
def predict_dl():
    try:
        data = request.get_json()
        values = [[data[field] for field in fields]]
        df = pd.DataFrame(values, columns=fields)
        scaled = dl_scaler.transform(df)
        tensor_input = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = dl_model(tensor_input).item()
            prediction = int(prob > 0.5)
        return jsonify({
            "model": "lion_DL",
            "result": "Likely Diabetic" if prediction == 1 else "Not Diabetic",
            "probability": round(prob, 3),
            "raw": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🌐 Route 3 - Unified prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_type = data.get("model", "dl")  # default to lion

        values = [[data[field] for field in fields]]
        df = pd.DataFrame(values, columns=fields)

        if model_type == "ml":
            scaled = ml_scaler.transform(df)
            prediction = ml_model.predict(scaled)[0]
            return jsonify({
                "model": "traditional_ML",
                "result": "Likely Diabetic" if prediction == 1 else "Not Diabetic",
                "raw": int(prediction)
            })
        else:
            scaled = dl_scaler.transform(df)
            tensor_input = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                prob = dl_model(tensor_input).item()
                prediction = int(prob > 0.5)
            return jsonify({
                "model": "lion_DL",
                "result": "Likely Diabetic" if prediction == 1 else "Not Diabetic",
                "probability": round(prob, 3),
                "raw": prediction
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 👋 Hello test
@app.route("/", methods=["GET"])
def home():
    return "🔥 Diabetes Prediction API is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
