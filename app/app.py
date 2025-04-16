from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

# Define input schema
class BMIInput(BaseModel):
    height_inches: float
    weight_pounds: float

# Load model
model = joblib.load("models/model.pkl")

# Initialize app
app = FastAPI()

# Attach Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {"message": "BMI Predictor is running"}

@app.post("/predict")
def predict_bmi(data: BMIInput):
    height_m = data.height_inches * 0.0254
    weight_kg = data.weight_pounds * 0.453592
    bmi = model.predict(np.array([[height_m, weight_kg]]))[0]
    return {"predicted_bmi": round(bmi, 2)}
