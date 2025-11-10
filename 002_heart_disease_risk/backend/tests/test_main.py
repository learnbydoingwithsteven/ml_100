"""
Comprehensive tests for Heart Disease Risk Prediction API
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint returns API information"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "Heart Disease Risk Prediction API"
    assert data["version"] == "2.0.0"
    assert "endpoints" in data


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_status" in data
    assert "timestamp" in data


def test_predict_valid_patient():
    """Test prediction with valid patient data"""
    patient_data = {
        "age": 55,
        "resting_bp": 130,
        "cholesterol": 240,
        "max_heart_rate": 150,
        "st_depression": 1.2,
        "num_vessels": 1,
        "fasting_bs": 110,
        "bmi": 27.5
    }
    
    response = client.post("/api/v1/predict", json=patient_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_level" in data
    assert data["risk_level"] in ["low", "high"]
    assert "risk_score" in data
    assert data["risk_score"] in [0, 1]
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
    assert "timestamp" in data


def test_predict_high_risk_patient():
    """Test prediction for a high-risk patient"""
    # Patient with multiple risk factors
    patient_data = {
        "age": 75,
        "resting_bp": 180,
        "cholesterol": 350,
        "max_heart_rate": 100,
        "st_depression": 4.5,
        "num_vessels": 3,
        "fasting_bs": 180,
        "bmi": 35.0
    }
    
    response = client.post("/api/v1/predict", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    # Should be high risk given the extreme values
    assert data["probability"] > 0.5


def test_predict_low_risk_patient():
    """Test prediction for a low-risk patient"""
    # Young, healthy patient
    patient_data = {
        "age": 35,
        "resting_bp": 110,
        "cholesterol": 150,
        "max_heart_rate": 180,
        "st_depression": 0.1,
        "num_vessels": 0,
        "fasting_bs": 90,
        "bmi": 22.0
    }
    
    response = client.post("/api/v1/predict", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    # Should be low risk
    assert data["probability"] < 0.5


def test_predict_invalid_age():
    """Test prediction with invalid age (out of range)"""
    patient_data = {
        "age": 25,  # Too young
        "resting_bp": 130,
        "cholesterol": 240,
        "max_heart_rate": 150,
        "st_depression": 1.2,
        "num_vessels": 1,
        "fasting_bs": 110,
        "bmi": 27.5
    }
    
    response = client.post("/api/v1/predict", json=patient_data)
    assert response.status_code == 422  # Validation error


def test_predict_missing_field():
    """Test prediction with missing required field"""
    patient_data = {
        "age": 55,
        "resting_bp": 130,
        # Missing cholesterol
        "max_heart_rate": 150,
        "st_depression": 1.2,
        "num_vessels": 1,
        "fasting_bs": 110,
        "bmi": 27.5
    }
    
    response = client.post("/api/v1/predict", json=patient_data)
    assert response.status_code == 422  # Validation error


def test_batch_predict():
    """Test batch prediction with multiple patients"""
    patients = {
        "patients": [
            {
                "age": 55,
                "resting_bp": 130,
                "cholesterol": 240,
                "max_heart_rate": 150,
                "st_depression": 1.2,
                "num_vessels": 1,
                "fasting_bs": 110,
                "bmi": 27.5
            },
            {
                "age": 45,
                "resting_bp": 120,
                "cholesterol": 200,
                "max_heart_rate": 170,
                "st_depression": 0.5,
                "num_vessels": 0,
                "fasting_bs": 100,
                "bmi": 24.0
            }
        ]
    }
    
    response = client.post("/api/v1/predict/batch", json=patients)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    
    # Each prediction should have required fields
    for pred in data["predictions"]:
        assert "risk_level" in pred
        assert "probability" in pred


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "trained" in data
    assert data["trained"] == True
    assert "feature_names" in data
    assert len(data["feature_names"]) == 8
    assert "feature_importances" in data
    assert "metrics" in data


def test_feature_importances():
    """Test feature importances endpoint"""
    response = client.get("/api/v1/feature-importances")
    assert response.status_code == 200
    
    data = response.json()
    assert "feature_importances" in data
    assert len(data["feature_importances"]) == 8
    
    # Verify all expected features are present
    expected_features = [
        'age', 'resting_bp', 'cholesterol', 'max_heart_rate',
        'st_depression', 'num_vessels', 'fasting_bs', 'bmi'
    ]
    for feature in expected_features:
        assert feature in data["feature_importances"]


def test_retrain_model():
    """Test model retraining endpoint"""
    response = client.post("/api/v1/model/retrain?n_samples=500")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert "train_accuracy" in data["metrics"]
    assert "test_accuracy" in data["metrics"]
    
    # Verify metrics are reasonable
    assert 0.5 < data["metrics"]["train_accuracy"] < 1.0
    assert 0.5 < data["metrics"]["test_accuracy"] < 1.0


def test_prediction_consistency():
    """Test that predictions are consistent for the same input"""
    patient_data = {
        "age": 55,
        "resting_bp": 130,
        "cholesterol": 240,
        "max_heart_rate": 150,
        "st_depression": 1.2,
        "num_vessels": 1,
        "fasting_bs": 110,
        "bmi": 27.5
    }
    
    response1 = client.post("/api/v1/predict", json=patient_data)
    response2 = client.post("/api/v1/predict", json=patient_data)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    # Predictions should be identical
    assert data1["risk_level"] == data2["risk_level"]
    assert data1["probability"] == data2["probability"]
    assert data1["risk_score"] == data2["risk_score"]
