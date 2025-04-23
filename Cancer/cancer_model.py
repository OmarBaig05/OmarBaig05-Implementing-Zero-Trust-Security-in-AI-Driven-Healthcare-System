import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import hashlib
import logging
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()
MODEL_HASH = os.getenv("SHA_HASH")
if MODEL_HASH is None:
    raise ValueError("SHA_HASH environment variable not set.")

# Load model with integrity check
model_path = './Cancer/cancer_pred.pkl'

# Initialize FastAPI app
app = FastAPI(title="Secure Breast Cancer Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Define feature columns
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Feature ranges with 2% margin
feature_ranges = {
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

# Expected feature statistics (mean and std) for z-score validation
feature_stats = {f: {'mean': (r[0] + r[1]) / 2, 'std': (r[1] - r[0]) / 6} for f, r in feature_ranges.items()}

# Pydantic model for single prediction
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

# Pydantic model for batch predictions
class CancerDataBatch(BaseModel):
    data: List[CancerData]

def verify_model_integrity(model_path: str) -> bool:
    try:
        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash == MODEL_HASH
    except Exception as e:
        logger.error(f"Model integrity check failed: {str(e)}")
        return False


if not os.path.exists(model_path):
    logger.error("Model file not found")
    raise Exception("Model file not found. Ensure 'cancer_pred.pkl' exists.")
if not verify_model_integrity(model_path):
    logger.error("Model integrity verification failed")
    raise Exception("Model integrity check failed. Possible tampering detected.")
try:
    model = joblib.load(model_path)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")


# Adversarial input detection
def detect_adversarial_input(input_data: pd.DataFrame) -> bool:
    # Z-score check
    for col in input_data.columns:
        z_scores = np.abs((input_data[col] - feature_stats[col]['mean']) / feature_stats[col]['std'])
        if (z_scores > 3).any():
            logger.warning(f"High z-score detected in {col}")
            return True
    return False

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(data: CancerData, request: Request):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict(by_alias=True)], columns=feature_columns)
        
        # Check for adversarial input
        if detect_adversarial_input(input_data):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        # Predict
        prediction = model.predict(input_data)[0]
        prediction_int = 1 if prediction == 'M' else 0 if prediction == 'B' else int(prediction)
        
        logger.info(f"Prediction made for client {request.client.host}: {prediction_int}")
        return {"prediction": prediction_int}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
@limiter.limit("3/minute")
async def predict_batch(batch: CancerDataBatch, request: Request):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([item.dict(by_alias=True) for item in batch.data], columns=feature_columns)
        
        # Check for adversarial input
        if detect_adversarial_input(input_data):
            logger.warning(f"Adversarial input detected from {request.client.host}")
            raise HTTPException(status_code=400, detail="Suspicious input detected")
        
        # Predict
        predictions = model.predict(input_data)
        predictions_int = [1 if p == 'M' else 0 if p == 'B' else int(p) for p in predictions]
        
        logger.info(f"Batch prediction made for client {request.client.host}: {len(predictions_int)} predictions")
        return {"predictions": predictions_int}
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Secure Breast Cancer Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Dummy data for testing (within computed ranges)


# for /predict

# {
#   "radius_mean": 17.99,
#   "texture_mean": 10.38,
#   "perimeter_mean": 122.8,
#   "area_mean": 1001.0,
#   "smoothness_mean": 0.1184,
#   "compactness_mean": 0.2776,
#   "concavity_mean": 0.3001,
#   "concave points_mean": 0.1471,
#   "symmetry_mean": 0.2419,
#   "fractal_dimension_mean": 0.07871,
#   "radius_se": 1.095,
#   "texture_se": 0.9053,
#   "perimeter_se": 8.589,
#   "area_se": 153.4,
#   "smoothness_se": 0.006399,
#   "compactness_se": 0.04904,
#   "concavity_se": 0.05373,
#   "concave points_se": 0.01587,
#   "symmetry_se": 0.03003,
#   "fractal_dimension_se": 0.006193,
#   "radius_worst": 25.38,
#   "texture_worst": 17.33,
#   "perimeter_worst": 184.6,
#   "area_worst": 2019.0,
#   "smoothness_worst": 0.1622,
#   "compactness_worst": 0.6656,
#   "concavity_worst": 0.7119,
#   "concave points_worst": 0.2654,
#   "symmetry_worst": 0.4601,
#   "fractal_dimension_worst": 0.1189
# }


# for /predict_batch

# {
#   "data": [
#     {
#       "radius_mean": 13.54,
#       "texture_mean": 14.36,
#       "perimeter_mean": 87.46,
#       "area_mean": 566.3,
#       "smoothness_mean": 0.09779,
#       "compactness_mean": 0.08129,
#       "concavity_mean": 0.06664,
#       "concave points_mean": 0.04781,
#       "symmetry_mean": 0.1885,
#       "fractal_dimension_mean": 0.05766,
#       "radius_se": 0.2699,
#       "texture_se": 0.7886,
#       "perimeter_se": 2.058,
#       "area_se": 23.56,
#       "smoothness_se": 0.008462,
#       "compactness_se": 0.0146,
#       "concavity_se": 0.02387,
#       "concave points_se": 0.01315,
#       "symmetry_se": 0.0198,
#       "fractal_dimension_se": 0.0023,
#       "radius_worst": 15.11,
#       "texture_worst": 19.26,
#       "perimeter_worst": 99.7,
#       "area_worst": 711.2,
#       "smoothness_worst": 0.144,
#       "compactness_worst": 0.1773,
#       "concavity_worst": 0.239,
#       "concave points_worst": 0.1288,
#       "symmetry_worst": 0.2977,
#       "fractal_dimension_worst": 0.07259
#     },
#     {
#       "radius_mean": 11.84,
#       "texture_mean": 18.7,
#       "perimeter_mean": 77.93,
#       "area_mean": 440.6,
#       "smoothness_mean": 0.1109,
#       "compactness_mean": 0.1516,
#       "concavity_mean": 0.1218,
#       "concave points_mean": 0.05182,
#       "symmetry_mean": 0.2301,
#       "fractal_dimension_mean": 0.07799,
#       "radius_se": 0.4824,
#       "texture_se": 1.03,
#       "perimeter_se": 3.475,
#       "area_se": 41.0,
#       "smoothness_se": 0.005551,
#       "compactness_se": 0.03414,
#       "concavity_se": 0.04205,
#       "concave points_se": 0.01044,
#       "symmetry_se": 0.02273,
#       "fractal_dimension_se": 0.005667,
#       "radius_worst": 16.82,
#       "texture_worst": 28.12,
#       "perimeter_worst": 119.4,
#       "area_worst": 888.7,
#       "smoothness_worst": 0.1637,
#       "compactness_worst": 0.5775,
#       "concavity_worst": 0.6956,
#       "concave points_worst": 0.1546,
#       "symmetry_worst": 0.4761,
#       "fractal_dimension_worst": 0.1402
#     }
#   ]
# }



# for wrong values outpu will be like, they are referenced from feature_ranges: 
# {
#     "detail": [
#         {
#             "type": "less_than_equal",
#             "loc": [
#                 "body",
#                 "fractal_dimension_worst"
#             ],
#             "msg": "Input should be less than or equal to 0.21165",
#             "input": 0.21166,
#             "ctx": {
#                 "le": 0.21165
#             }
#         }
#     ]
# }