import pytest
from fastapi.testclient import TestClient
from app.main import app, initialize_model
import time

client = TestClient(app)

# Initialize model before running tests
@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Initialize model before all tests"""
    initialize_model()
    yield

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "Disease Diagnosis System"
    assert data["version"] == "2.0.0"
    assert "model_trained" in data

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_ready" in data
    assert "timestamp" in data

def test_diagnose_healthy_patient():
    """Test diagnosis with healthy patient data"""
    patient_data = {
        "body_temperature": 37.0,
        "blood_pressure": 120.0,
        "heart_rate": 70.0,
        "glucose_level": 90.0,
        "cholesterol": 180.0,
        "bmi": 22.0,
        "age": 30.0,
        "symptom_severity": 10.0
    }
    
    response = client.post("/api/v1/diagnose", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    assert "diagnosis" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "risk_factors" in data
    assert "recommendations" in data
    assert data["confidence"] >= 0.0 and data["confidence"] <= 1.0

def test_diagnose_critical_patient():
    """Test diagnosis with critical patient data"""
    patient_data = {
        "body_temperature": 40.0,
        "blood_pressure": 180.0,
        "heart_rate": 140.0,
        "glucose_level": 250.0,
        "cholesterol": 300.0,
        "bmi": 35.0,
        "age": 70.0,
        "symptom_severity": 90.0
    }
    
    response = client.post("/api/v1/diagnose", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    assert data["diagnosis_code"] >= 2  # Should be moderate or higher
    assert len(data["risk_factors"]) > 1  # Should have multiple risk factors

def test_diagnose_invalid_data():
    """Test diagnosis with invalid patient data"""
    invalid_data = {
        "body_temperature": 50.0,  # Too high
        "blood_pressure": 120.0,
        "heart_rate": 70.0,
        "glucose_level": 90.0,
        "cholesterol": 180.0,
        "bmi": 22.0,
        "age": 30.0,
        "symptom_severity": 10.0
    }
    
    response = client.post("/api/v1/diagnose", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_train_model():
    """Test model training endpoint"""
    response = client.post("/api/v1/train")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "model_accuracy" in data
    assert data["model_accuracy"] > 0.5  # Should be better than random
    assert "training_samples" in data
    assert "test_samples" in data

def test_model_info():
    """Test model information endpoint"""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "RandomForestClassifier"
    assert "features" in data
    assert len(data["features"]) == 8
    assert "disease_labels" in data

def test_feature_importance():
    """Test feature importance endpoint"""
    response = client.get("/api/v1/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    assert "most_important" in data
    assert "least_important" in data
    assert len(data["feature_importance"]) == 8

def test_diagnose_edge_cases():
    """Test diagnosis with edge case values"""
    edge_cases = [
        {
            "body_temperature": 35.0,
            "blood_pressure": 60.0,
            "heart_rate": 40.0,
            "glucose_level": 50.0,
            "cholesterol": 100.0,
            "bmi": 10.0,
            "age": 0.0,
            "symptom_severity": 0.0
        },
        {
            "body_temperature": 42.0,
            "blood_pressure": 200.0,
            "heart_rate": 200.0,
            "glucose_level": 400.0,
            "cholesterol": 400.0,
            "bmi": 60.0,
            "age": 120.0,
            "symptom_severity": 100.0
        }
    ]
    
    for patient_data in edge_cases:
        response = client.post("/api/v1/diagnose", json=patient_data)
        assert response.status_code == 200
        data = response.json()
        assert "diagnosis" in data
