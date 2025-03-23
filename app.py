from fastapi import FastAPI
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running!"}

@app.post("/predict/")
def predict(features: dict):
    # Convert input dictionary to DataFrame
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    input_data = pd.DataFrame([features], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return {"predicted_crop": prediction[0]}
