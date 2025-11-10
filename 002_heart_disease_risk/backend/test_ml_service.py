"""
Quick test to debug ml_service
"""

from app.ml_service import HeartDiseaseModel
import traceback

try:
    print("Creating model...")
    model = HeartDiseaseModel()
    print("Model created successfully!")
    
    print("\nTesting prediction...")
    features = {
        "age": 55,
        "resting_bp": 130,
        "cholesterol": 240,
        "max_heart_rate": 150,
        "st_depression": 1.2,
        "num_vessels": 1,
        "fasting_bs": 110,
        "bmi": 27.5
    }
    
    result = model.predict(features)
    print(f"Prediction successful: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
