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
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
MODEL_HASH_CANCER = os.getenv("SHA_HASH")
MODEL_HASH_HEART = os.getenv("SHA_HASH_256_HEART_FAILURE_MODEL")
SCALER_HASH_HEART = os.getenv("SHA_HASH_256_SCALER")
allowed_origin = os.getenv("ALLOWED_ORIGIN", "").strip()
print(f"Allowed origin: {allowed_origin}")

if MODEL_HASH_CANCER is None or MODEL_HASH_HEART is None or SCALER_HASH_HEART is None:
    raise ValueError("Required environment variables not set.")

# Initialize FastAPI app
app = FastAPI(title="Secure Medical Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],  # Only allow localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log unauthorized origins
class LogUnauthorizedOriginsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")
        logger.info(f"Request origin: {origin}")
        
        # Block requests without an origin header or from unauthorized origins
        if not origin or origin != allowed_origin:
            logger.warning(f"Unauthorized request from origin: {origin}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Requests from this origin are not allowed."},
            )
        
        return await call_next(request)

# Add the custom middleware
app.add_middleware(LogUnauthorizedOriginsMiddleware)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Breast Cancer Prediction Setup ---

# Define cancer feature columns
cancer_feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Cancer feature ranges
cancer_feature_ranges = {
    'radius_mean': (6.84138, 28.6722),
    'texture_mean': (9.5158, 40.0656),
    'perimeter_mean': (42.9142, 192.27),
    'area_mean': (140.63, 2551.02),
    'smoothness_mean': (0.0515764, 0.166668),
    'compactness_mean': (0.0189924, 0.352308),
    'concavity_mean': (0.0, 0.435336),
    'concave points_mean': (0.0, 0.205404),
    'symmetry_mean': (0.10388, 0.31008),
    'fractal_dimension_mean': (0.049592, 0.095796),
    'radius_se': (0.10976, 2.86536),
    'texture_se': (0.29686, 4.40494),
    'perimeter_se': (0.66924, 21.7114),
    'area_se': (6.3604, 404.632),
    'smoothness_se': (0.00225484, 0.0313496),
    'compactness_se': (0.0023468, 0.135796),
    'concavity_se': (0.0, 0.408612),
    'concave points_se': (0.0, 0.0528792),
    'symmetry_se': (0.0078092, 0.079948),
    'fractal_dimension_se': (0.00077784, 0.029868),
    'radius_worst': (7.146, 36.414),
    'texture_worst': (11.7796, 50.5308),
    'perimeter_worst': (49.4018, 256.224),
    'area_worst': (181.496, 4339.08),
    'smoothness_worst': (0.0697466, 0.227052),
    'compactness_worst': (0.0267442, 1.07916),
    'concavity_worst': (0.0, 1.27704),
    'concave points_worst': (0.0, 0.29682),
    'symmetry_worst': (0.15337, 0.677076),
    'fractal_dimension_worst': (0.0539392, 0.21165)
}

cancer_feature_stats = {f: {'mean': (r[0] + r[1]) / 2, 'std': (r[1] - r[0]) / 6} for f, r in cancer_feature_ranges.items()}

# Pydantic model for cancer single prediction
class CancerData(BaseModel):
    radius_mean: float = Field(..., ge=6.84138, le=28.6722)
    texture_mean: float = Field(..., ge=9.5158, le=40.0656)
    perimeter_mean: float = Field(..., ge=42.9142, le=192.27)
    area_mean: float = Field(..., ge=140.63, le=2551.02)
    smoothness_mean: float = Field(..., ge=0.0515764, le=0.166668)
    compactness_mean: float = Field(..., ge=0.0189924, le=0.352308)
    concavity_mean: float = Field(..., ge=0.0, le=0.435336)
    concave_points_mean: float = Field(..., ge=0.0, le=0.205404, alias='concave points_mean')
    symmetry_mean: float = Field(..., ge=0.10388, le=0.31008)
    fractal_dimension_mean: float = Field(..., ge=0.049592, le=0.095796)
    radius_se: float = Field(..., ge=0.10976, le=2.86536)
    texture_se: float = Field(..., ge=0.29686, le=4.40494)
    perimeter_se: float = Field(..., ge=0.66924, le=21.7114)
    area_se: float = Field(..., ge=6.3604, le=404.632)
    smoothness_se: float = Field(..., ge=0.00225484, le=0.0313496)
    compactness_se: float = Field(..., ge=0.0023468, le=0.135796)
    concavity_se: float = Field(..., ge=0.0, le=0.408612)
    concave_points_se: float = Field(..., ge=0.0, le=0.0528792, alias='concave points_se')
    symmetry_se: float = Field(..., ge=0.0078092, le=0.079948)
    fractal_dimension_se: float = Field(..., ge=0.00077784, le=0.029868)
    radius_worst: float = Field(..., ge=7.146, le=36.414)
    texture_worst: float = Field(..., ge=11.7796, le=50.5308)
    perimeter_worst: float = Field(..., ge=49.4018, le=256.224)
    area_worst: float = Field(..., ge=181.496, le=4339.08)
    smoothness_worst: float = Field(..., ge=0.0697466, le=0.227052)
    compactness_worst: float = Field(..., ge=0.0267442, le=1.07916)
    concavity_worst: float = Field(..., ge=0.0, le=1.27704)
    concave_points_worst: float = Field(..., ge=0.0, le=0.29682, alias='concave points_worst')
    symmetry_worst: float = Field(..., ge=0.15337, le=0.677076)
    fractal_dimension_worst: float = Field(..., ge=0.0539392, le=0.21165)

# Pydantic model for cancer batch predictions
class CancerDataBatch(BaseModel):
    data: List[CancerData]

# Load cancer model
cancer_model_path = './Cancer/cancer_pred.pkl'

def verify_model_integrity(file_path: str, expected_hash: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        logger.error(f"Model integrity check failed: {str(e)}")
        return False

if not os.path.exists(cancer_model_path):
    logger.error("Cancer model file not found")
    raise Exception("Cancer model file not found. Ensure 'cancer_pred.pkl' exists.")
if not verify_model_integrity(cancer_model_path, MODEL_HASH_CANCER):
    logger.error("Cancer model integrity verification failed")
    raise Exception("Cancer model integrity check failed. Possible tampering detected.")
try:
    cancer_model = joblib.load(cancer_model_path)
except Exception as e:
    logger.error(f"Failed to load cancer model: {str(e)}")
    raise Exception(f"Failed to load cancer model: {str(e)}")

# --- Heart Failure Prediction Setup ---

# Define heart feature columns
heart_feature_columns = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
    'smoking', 'time'
]

# Heart feature ranges
heart_feature_ranges = {
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

heart_feature_stats = {f: {'mean': (r[0] + r[1]) / 2, 'std': (r[1] - r[0]) / 6} for f, r in heart_feature_ranges.items()}

# Pydantic model for heart single prediction
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

# Pydantic model for heart batch predictions
class HeartDataBatch(BaseModel):
    data: List[HeartData]

# Load heart model and scaler
heart_model_path = './Heart Disease/heart_failure_model.pkl'
heart_scaler_path = './Heart Disease/scaler.pkl'

if not os.path.exists(heart_model_path):
    logger.error("Heart model file not found")
    raise Exception(f"Heart model file not found. Ensure '{heart_model_path}' exists.")
if not os.path.exists(heart_scaler_path):
    logger.error("Heart scaler file not found")
    raise Exception(f"Heart scaler file not found. Ensure '{heart_scaler_path}' exists.")

if not verify_model_integrity(heart_model_path, MODEL_HASH_HEART):
    logger.error("Heart model integrity verification failed")
    raise Exception("Heart model integrity check failed. Possible tampering detected.")
if not verify_model_integrity(heart_scaler_path, SCALER_HASH_HEART):
    logger.error("Heart scaler integrity verification failed")
    raise Exception("Heart scaler integrity check failed. Possible tampering detected.")

try:
    heart_model = joblib.load(heart_model_path)
    heart_scaler = joblib.load(heart_scaler_path)
except Exception as e:
    logger.error(f"Failed to load heart model or scaler: {str(e)}")
    raise Exception(f"Failed to load heart model or scaler: {str(e)}")

# Adversarial input detection
def detect_adversarial_input(input_data: pd.DataFrame, feature_stats: dict) -> bool:
    for col in input_data.columns:
        z_scores = np.abs((input_data[col] - feature_stats[col]['mean']) / feature_stats[col]['std'])
        if (z_scores > 3).any():
            logger.warning(f"High z-score detected in {col}")
            return True
    return False

# --- Cancer Prediction Routes ---

@app.post("/cancer/predict")
@limiter.limit("5/minute")
async def predict_cancer(data: CancerData, request: Request):
    try:
        input_data = pd.DataFrame([data.dict(by_alias=True)], columns=cancer_feature_columns)
        
        if detect_adversarial_input(input_data, cancer_feature_stats):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        prediction = cancer_model.predict(input_data)[0]
        probability = cancer_model.predict_proba(input_data)[0].tolist()
        prediction_int = 1 if prediction == 'M' else 0 if prediction == 'B' else int(prediction)
        
        logger.info(f"Cancer prediction made for client {request.client.host}: {prediction_int}")
        return {
            "prediction": prediction_int,
            "probability": probability
        }
    except Exception as e:
        logger.error(f"Cancer prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cancer/predict_batch")
@limiter.limit("3/minute")
async def predict_cancer_batch(batch: CancerDataBatch, request: Request):
    try:
        input_data = pd.DataFrame([item.dict(by_alias=True) for item in batch.data], columns=cancer_feature_columns)
        
        if detect_adversarial_input(input_data, cancer_feature_stats):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        predictions = cancer_model.predict(input_data)
        probabilities = cancer_model.predict_proba(input_data).tolist()
        predictions_int = [1 if p == 'M' else 0 if p == 'B' else int(p) for p in predictions]
        
        logger.info(f"Cancer batch prediction made for client {request.client.host}: {len(predictions_int)} predictions")
        return {
            "predictions": predictions_int,
            "probabilities": probabilities
        }
    except Exception as e:
        logger.error(f"Cancer batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# --- Heart Failure Prediction Routes ---

@app.post("/heart/predict")
@limiter.limit("5/minute")
async def predict_heart(data: HeartData, request: Request):
    try:
        input_data = pd.DataFrame([data.dict()], columns=heart_feature_columns)
        
        if detect_adversarial_input(input_data, heart_feature_stats):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        input_scaled = heart_scaler.transform(input_data)
        prediction = heart_model.predict(input_scaled)[0]
        probability = heart_model.predict_proba(input_scaled)[0].tolist()
        
        logger.info(f"Heart prediction made for client {request.client.host}: {prediction}")
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        logger.error(f"Heart prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/heart/predict_batch")
@limiter.limit("3/minute")
async def predict_heart_batch(batch: HeartDataBatch, request: Request):
    try:
        input_data = pd.DataFrame([item.dict() for item in batch.data], columns=heart_feature_columns)
        
        if detect_adversarial_input(input_data, heart_feature_stats):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        input_scaled = heart_scaler.transform(input_data)
        predictions = heart_model.predict(input_scaled).tolist()
        probabilities = heart_model.predict_proba(input_scaled).tolist()
        
        logger.info(f"Heart batch prediction made for client {request.client.host}: {len(predictions)} predictions")
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
    except Exception as e:
        logger.error(f"Heart batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Secure Medical Prediction API (Breast Cancer & Heart Failure)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)