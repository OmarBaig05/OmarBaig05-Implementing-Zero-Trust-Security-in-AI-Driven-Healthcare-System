import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn



# FastAPI app
app = FastAPI(title="Heart Failure Prediction API")

# Pydantic model for single prediction
class HeartData(BaseModel):
    age: float = Field(..., ge=40.0, le=95.0, description="Age in years")
    anaemia: int = Field(..., ge=0, le=1, description="Anaemia (0 or 1)")
    creatinine_phosphokinase: int = Field(..., ge=23, le=7861, description="CPK level in mcg/L")
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes (0 or 1)")
    ejection_fraction: int = Field(..., ge=14, le=80, description="Ejection fraction percentage")
    high_blood_pressure: int = Field(..., ge=0, le=1, description="High blood pressure (0 or 1)")
    platelets: float = Field(..., ge=25100.0, le=850000.0, description="Platelets in kiloplatelets/mL")
    serum_creatinine: float = Field(..., ge=0.5, le=9.4, description="Serum creatinine in mg/dL")
    serum_sodium: int = Field(..., ge=113, le=148, description="Serum sodium in mEq/L")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 or 1)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking (0 or 1)")
    time: int = Field(..., ge=4, le=285, description="Follow-up period in days")

# Pydantic model for batch predictions
class HeartDataBatch(BaseModel):
    data: List[HeartData]

# Load model and scaler
model = joblib.load('heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
async def predict(data: HeartData):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0].tolist()
        
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(batch: HeartDataBatch):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([item.dict() for item in batch.data])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Predict
        predictions = model.predict(input_scaled).tolist()
        probabilities = model.predict_proba(input_scaled).tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Heart Failure Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



# dummy HeartData

# for /predict

{
    "age": 60.0,
    "anaemia": 0,
    "creatinine_phosphokinase": 250,
    "diabetes": 1,
    "ejection_fraction": 38,
    "high_blood_pressure": 0,
    "platelets": 262000.0,
    "serum_creatinine": 1.1,
    "serum_sodium": 137,
    "sex": 1,
    "smoking": 0,
    "time": 115
  }

# for /predict_batch

# {
#   "data": [
#     {
#       "age": 55.0,
#       "anaemia": 1,
#       "creatinine_phosphokinase": 582,
#       "diabetes": 0,
#       "ejection_fraction": 30,
#       "high_blood_pressure": 1,
#       "platelets": 212500.0,
#       "serum_creatinine": 0.9,
#       "serum_sodium": 134,
#       "sex": 0,
#       "smoking": 0,
#       "time": 73
#     },
#     {
#       "age": 70.0,
#       "anaemia": 0,
#       "creatinine_phosphokinase": 116,
#       "diabetes": 1,
#       "ejection_fraction": 45,
#       "high_blood_pressure": 0,
#       "platelets": 303500.0,
#       "serum_creatinine": 1.4,
#       "serum_sodium": 140,
#       "sex": 1,
#       "smoking": 1,
#       "time": 203
#     }
#   ]
# }