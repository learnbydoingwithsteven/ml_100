"""
Disease Diagnosis Application - Backend API
Machine Learning powered disease diagnosis system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Diagnosis System",
    description="ML-powered disease diagnosis with patient symptom analysis",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease categories
DISEASE_LABELS = {
    0: "Healthy",
    1: "Mild Condition",
    2: "Moderate Condition",
    3: "Severe Condition",
    4: "Critical Condition"
}

# Feature names (patient symptoms/metrics)
FEATURE_NAMES = [
    "body_temperature",
    "blood_pressure",
    "heart_rate",
    "glucose_level",
    "cholesterol",
    "bmi",
    "age",
    "symptom_severity"
]

# Global model storage
model = None
model_trained = False
training_history = []

class PatientData(BaseModel):
    body_temperature: float = Field(ge=35.0, le=42.0, description="Body temperature in Celsius")
    blood_pressure: float = Field(ge=60.0, le=200.0, description="Systolic blood pressure")
    heart_rate: float = Field(ge=40.0, le=200.0, description="Heart rate in BPM")
    glucose_level: float = Field(ge=50.0, le=400.0, description="Blood glucose level")
    cholesterol: float = Field(ge=100.0, le=400.0, description="Cholesterol level")
    bmi: float = Field(ge=10.0, le=60.0, description="Body Mass Index")
    age: float = Field(ge=0.0, le=120.0, description="Patient age")
    symptom_severity: float = Field(ge=0.0, le=100.0, description="Symptom severity score")

class DiagnosisResponse(BaseModel):
    diagnosis: str
    diagnosis_code: int
    confidence: float
    probabilities: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    timestamp: datetime

class TrainingResponse(BaseModel):
    status: str
    model_accuracy: float
    training_samples: int
    test_samples: int
    timestamp: datetime

def generate_training_data(n_samples: int = 2000):
    """Generate synthetic training data for disease diagnosis"""
    np.random.seed(42)
    
    feature_1 = np.random.uniform(36, 41, n_samples)  # body_temperature
    feature_2 = np.random.uniform(80, 180, n_samples)  # blood_pressure
    feature_3 = np.random.uniform(50, 150, n_samples)  # heart_rate
    feature_4 = np.random.uniform(70, 200, n_samples)  # glucose_level
    feature_5 = np.random.uniform(150, 300, n_samples)  # cholesterol
    feature_6 = np.random.uniform(18, 40, n_samples)  # bmi
    feature_7 = np.random.uniform(18, 85, n_samples)  # age
    feature_8 = np.random.uniform(0, 100, n_samples)  # symptom_severity
    
    # Generate target based on weighted combination of features
    score = (
        feature_1 * 0.3 +  # temperature weight
        feature_2 * 0.15 +  # blood pressure weight
        feature_3 * 0.15 +  # heart rate weight
        feature_4 * 0.1 +  # glucose weight
        feature_8 * 0.3  # symptom severity weight
    )
    
    percentiles = np.linspace(0, 100, 6)
    target = np.digitize(score, np.percentile(score, percentiles[1:-1]))
    
    df = pd.DataFrame({
        'body_temperature': feature_1,
        'blood_pressure': feature_2,
        'heart_rate': feature_3,
        'glucose_level': feature_4,
        'cholesterol': feature_5,
        'bmi': feature_6,
        'age': feature_7,
        'symptom_severity': feature_8,
        'target': target
    })
    
    return df

def get_risk_factors(patient_data: PatientData) -> List[str]:
    """Identify risk factors based on patient data"""
    risk_factors = []
    
    if patient_data.body_temperature > 38.5:
        risk_factors.append("Elevated body temperature")
    if patient_data.blood_pressure > 140:
        risk_factors.append("High blood pressure")
    if patient_data.heart_rate > 100:
        risk_factors.append("Elevated heart rate")
    if patient_data.glucose_level > 140:
        risk_factors.append("High blood glucose")
    if patient_data.cholesterol > 240:
        risk_factors.append("High cholesterol")
    if patient_data.bmi > 30:
        risk_factors.append("Obesity (BMI > 30)")
    if patient_data.age > 65:
        risk_factors.append("Advanced age")
    if patient_data.symptom_severity > 70:
        risk_factors.append("Severe symptoms")
    
    return risk_factors if risk_factors else ["No significant risk factors detected"]

def get_recommendations(diagnosis_code: int, risk_factors: List[str]) -> List[str]:
    """Generate recommendations based on diagnosis"""
    recommendations = []
    
    if diagnosis_code == 0:
        recommendations.append("Maintain healthy lifestyle")
        recommendations.append("Regular check-ups recommended")
    elif diagnosis_code == 1:
        recommendations.append("Monitor symptoms closely")
        recommendations.append("Consult with primary care physician")
        recommendations.append("Stay hydrated and rest")
    elif diagnosis_code == 2:
        recommendations.append("Medical consultation recommended")
        recommendations.append("Follow prescribed treatment plan")
        recommendations.append("Monitor vital signs daily")
    elif diagnosis_code >= 3:
        recommendations.append("Immediate medical attention required")
        recommendations.append("Follow emergency protocols")
        recommendations.append("Hospitalization may be necessary")
    
    return recommendations

def initialize_model():
    """Initialize and train the model"""
    global model, model_trained
    try:
        logger.info("Training disease diagnosis model...")
        df = generate_training_data()
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        model_trained = True
        
        logger.info(f"Model trained successfully! Accuracy: {accuracy:.4f}")
        return True
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        model_trained = False
        return False

@app.on_event("startup")
async def startup_event():
    """Train model on startup"""
    initialize_model()

@app.get("/")
async def root():
    return {
        "app": "Disease Diagnosis System",
        "status": "operational",
        "version": "2.0.0",
        "model_trained": model_trained
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_ready": model_trained,
        "timestamp": datetime.now()
    }

@app.post("/api/v1/diagnose", response_model=DiagnosisResponse)
async def diagnose(patient_data: PatientData):
    """Diagnose disease based on patient symptoms and metrics"""
    if not model_trained or model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        # Prepare input data
        input_data = np.array([[
            patient_data.body_temperature,
            patient_data.blood_pressure,
            patient_data.heart_rate,
            patient_data.glucose_level,
            patient_data.cholesterol,
            patient_data.bmi,
            patient_data.age,
            patient_data.symptom_severity
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get risk factors and recommendations
        risk_factors = get_risk_factors(patient_data)
        recommendations = get_recommendations(prediction, risk_factors)
        
        # Build probability dictionary
        prob_dict = {DISEASE_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return DiagnosisResponse(
            diagnosis=DISEASE_LABELS[prediction],
            diagnosis_code=int(prediction),
            confidence=float(probabilities[prediction]),
            probabilities=prob_dict,
            risk_factors=risk_factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/train", response_model=TrainingResponse)
async def train_model():
    """Retrain the model with fresh data"""
    global model, model_trained
    
    try:
        df = generate_training_data()
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        model_trained = True
        
        training_history.append({
            "timestamp": datetime.now(),
            "accuracy": accuracy
        })
        
        return TrainingResponse(
            status="success",
            model_accuracy=accuracy,
            training_samples=len(X_train),
            test_samples=len(X_test),
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/model-info")
async def model_info():
    """Get information about the current model"""
    if not model_trained or model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features": FEATURE_NAMES,
        "disease_labels": DISEASE_LABELS,
        "training_history": training_history[-5:] if training_history else []
    }

@app.get("/api/v1/feature-importance")
async def feature_importance():
    """Get feature importance from the trained model"""
    if not model_trained or model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        importance = model.feature_importances_
        feature_importance = {
            name: float(imp) 
            for name, imp in zip(FEATURE_NAMES, importance)
        }
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "feature_importance": dict(sorted_features),
            "most_important": sorted_features[0][0],
            "least_important": sorted_features[-1][0]
        }
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
