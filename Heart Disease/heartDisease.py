import pandas as pd
import numpy as np
import joblib
import hashlib
import logging
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import dotenv
import os

dotenv.load_dotenv()
# Load environment variables
SHA_HASH_256_HEART_FAILURE_MODEL = os.getenv("SHA_HASH_256_HEART_FAILURE_MODEL")
SHA_HASH_256_SCALER = os.getenv("SHA_HASH_256_SCALER")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Secure Heart Failure Prediction API")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define feature columns
feature_columns = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
    'smoking', 'time'
]

# Feature ranges
feature_ranges = {
    'age': (40.0, 95.0),
    'anaemia': (0, 1),
    'creatinine_phosphokinase': (23, 7861),
    'diabetes': (0, 1),
    'ejection_fraction': (14, 80),
    'high_blood_pressure': (0, 1),
    'platelets': (25100.0, 850000.0),
    'serum_creatinine': (0.5, 9.4),
    'serum_sodium': (113, 148),
    'sex': (0, 1),
    'smoking': (0, 1),
    'time': (4, 285)
}

# Expected feature statistics for z-score validation
feature_stats = {f: {'mean': (r[0] + r[1]) / 2, 'std': (r[1] - r[0]) / 6} for f, r in feature_ranges.items()}

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

# Model and scaler paths and hashes
MODEL_PATH = './Heart Disease/heart_failure_model.pkl'
SCALER_PATH = './Heart Disease/scaler.pkl'
MODEL_HASH = SHA_HASH_256_HEART_FAILURE_MODEL
SCALER_HASH = SHA_HASH_256_SCALER

# Verify file integrity
def verify_file_integrity(file_path: str, expected_hash: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        logger.error(f"Integrity check failed for {file_path}: {str(e)}")
        return False

# Load model and scaler with integrity checks
if not os.path.exists(MODEL_PATH):
    logger.error("Model file not found")
    raise Exception(f"Model file not found. Ensure '{MODEL_PATH}' exists.")
if not os.path.exists(SCALER_PATH):
    logger.error("Scaler file not found")
    raise Exception(f"Scaler file not found. Ensure '{SCALER_PATH}' exists.")

if not verify_file_integrity(MODEL_PATH, MODEL_HASH):
    logger.error("Model integrity verification failed")
    raise Exception("Model integrity check failed. Possible tampering detected.")
if not verify_file_integrity(SCALER_PATH, SCALER_HASH):
    logger.error("Scaler integrity verification failed")
    raise Exception("Scaler integrity check failed. Possible tampering detected.")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    logger.error(f"Failed to load model or scaler: {str(e)}")
    raise Exception(f"Failed to load model or scaler: {str(e)}")

# Adversarial input detection
def detect_adversarial_input(input_data: pd.DataFrame) -> bool:
    for col in input_data.columns:
        z_scores = np.abs((input_data[col] - feature_stats[col]['mean']) / feature_stats[col]['std'])
        if (z_scores > 3).any():
            logger.warning(f"High z-score detected in {col}")
            return True
    return False

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(data: HeartData, request: Request):
    try:
        input_data = pd.DataFrame([data.dict()], columns=feature_columns)
        
        if detect_adversarial_input(input_data):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0].tolist()
        
        logger.info(f"Prediction made for client {request.client.host}: {prediction}")
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
@limiter.limit("3/minute")
async def predict_batch(batch: HeartDataBatch, request: Request):
    try:
        input_data = pd.DataFrame([item.dict() for item in batch.data], columns=feature_columns)
        
        if detect_adversarial_input(input_data):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        input_scaled = scaler.transform(input_data)
        predictions = model.predict(input_scaled).tolist()
        probabilities = model.predict_proba(input_scaled).tolist()
        
        logger.info(f"Batch prediction made for client {request.client.host}: {len(predictions)} predictions")
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Secure Heart Failure Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# dummy HeartData

# for /predict

# {
#     "age": 60.0,
#     "anaemia": 0,
#     "creatinine_phosphokinase": 250,
#     "diabetes": 1,
#     "ejection_fraction": 38,
#     "high_blood_pressure": 0,
#     "platelets": 262000.0,
#     "serum_creatinine": 1.1,
#     "serum_sodium": 137,
#     "sex": 1,
#     "smoking": 0,
#     "time": 115
#   }

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



# Sample output for single prediction (correct one)
# {
#     "prediction": 0,
#     "probability": [
#         0.859808697492288,
#         0.140191302507712
#     ]
# }

# For wrong values
# {
#     "detail": [
#         {
#             "type": "less_than_equal",
#             "loc": [
#                 "body",
#                 "serum_sodium"
#             ],
#             "msg": "Input should be less than or equal to 148",
#             "input": 500,
#             "ctx": {
#                 "le": 148
#             }
#         }
#     ]
# }