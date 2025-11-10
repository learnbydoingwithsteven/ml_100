"""
Heart Disease Risk Prediction API - Backend
Production-ready ML application for heart disease risk assessment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime
import logging
from app.ml_service import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Machine Learning API for predicting heart disease risk based on patient data",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    """Patient data for heart disease risk prediction"""
    age: float = Field(..., ge=30, le=80, description="Age in years (30-80)")
    resting_bp: float = Field(..., ge=90, le=200, description="Resting blood pressure in mmHg (90-200)")
    cholesterol: float = Field(..., ge=100, le=400, description="Cholesterol level in mg/dl (100-400)")
    max_heart_rate: float = Field(..., ge=60, le=200, description="Maximum heart rate in bpm (60-200)")
    st_depression: float = Field(..., ge=0, le=6, description="ST depression induced by exercise (0-6)")
    num_vessels: int = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0-3)")
    fasting_bs: float = Field(..., ge=70, le=200, description="Fasting blood sugar in mg/dl (70-200)")
    bmi: float = Field(..., ge=18, le=45, description="Body Mass Index (18-45)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55,
                "resting_bp": 130,
                "cholesterol": 240,
                "max_heart_rate": 150,
                "st_depression": 1.2,
                "num_vessels": 1,
                "fasting_bs": 110,
                "bmi": 27.5
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    risk_level: str
    risk_score: int
    probability: float
    confidence: float
    features: Dict
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    patients: List[PatientData]


class ModelInfo(BaseModel):
    """Model information response"""
    trained: bool
    feature_names: List[str]
    feature_importances: Dict[str, float]
    metrics: Dict
    timestamp: datetime


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": "Heart Disease Risk Prediction API",
        "status": "operational",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "model_info": "/api/v1/model/info",
            "retrain": "/api/v1/model/retrain"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        model = get_model()
        model_status = "trained" if model.model is not None else "not_trained"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now()
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict heart disease risk for a single patient
    
    Returns:
        - risk_level: 'low' or 'high'
        - risk_score: 0 (low) or 1 (high)
        - probability: Probability of high risk (0-1)
        - confidence: Confidence in prediction (0-1)
    """
    try:
        model = get_model()
        
        # Convert PatientData to dict
        features = patient.model_dump()
        
        # Make prediction
        result = model.predict(features)
        
        return PredictionResponse(
            **result,
            timestamp=datetime.now()
        )
    except Exception as e:
        import traceback
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict heart disease risk for multiple patients
    
    Returns a list of predictions
    """
    try:
        model = get_model()
        
        # Convert to list of dicts
        patients = [p.model_dump() for p in request.patients]
        
        # Make batch prediction
        results = model.predict_batch(patients)
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/api/v1/model/info", response_model=ModelInfo)
async def model_info():
    """
    Get information about the trained model
    
    Returns model metrics, feature importances, and training info
    """
    try:
        model = get_model()
        info = model.get_model_info()
        
        return ModelInfo(
            **info,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/api/v1/model/retrain")
async def retrain_model(n_samples: int = 2000):
    """
    Retrain the model with new synthetic data
    
    Args:
        n_samples: Number of samples to generate for training
    
    Returns training metrics
    """
    try:
        model = get_model()
        metrics = model.train(n_samples=n_samples)
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/api/v1/feature-importances")
async def feature_importances():
    """Get feature importances from the trained model"""
    try:
        model = get_model()
        importances = model.get_feature_importances()
        
        # Sort by importance
        sorted_importances = dict(sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return {
            "feature_importances": sorted_importances,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importances: {str(e)}")
