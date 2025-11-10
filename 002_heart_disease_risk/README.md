# Heart Disease Risk Prediction

## Overview
Advanced machine learning application for predicting heart disease risk using **Gradient Boosting** classification with realistic medical indicators.

## Problem Type
**Binary Classification** (Low Risk / High Risk)

## Features

### Standalone Application (`app.py`)
- **Realistic synthetic data generation** based on medical risk factors
- **Optimized Gradient Boosting model** with hyperparameter tuning
- **8 medical features**: age, blood pressure, cholesterol, heart rate, ST depression, vessel count, blood sugar, BMI
- **Advanced visualizations**: 6-panel dashboard with ROC curve, PR curve, confusion matrix
- **Cross-validation** for robust performance estimation
- **Model persistence** with joblib
- **Comprehensive metrics**: accuracy, precision, recall, F1, ROC-AUC

### Production API (`backend/`)
- **FastAPI REST API** with interactive Swagger docs
- **Single & batch predictions** with validation
- **Model management** endpoints (info, retrain, feature importances)
- **Health checks** and monitoring
- **12 comprehensive tests** with 100% pass rate
- **Production-ready** with proper error handling

## Requirements

### Standalone App
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Backend API
```bash
cd backend
pip install -r requirements.txt
```

## Usage

### Standalone Application
```bash
python app.py
```

**Output Files:**
- `results.png` - 6-panel visualization dashboard (ROC, PR curve, confusion matrix, etc.)
- `results.txt` - Detailed classification metrics and feature importances
- `heart_disease_model.pkl` - Trained model for deployment
- `sample_predictions.csv` - Sample predictions for testing

### Backend API

**Start the server:**
```bash
cd backend
uvicorn app.main:app --reload
```

**Access the API:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

**Example prediction:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "resting_bp": 130,
    "cholesterol": 240,
    "max_heart_rate": 150,
    "st_depression": 1.2,
    "num_vessels": 1,
    "fasting_bs": 110,
    "bmi": 27.5
  }'
```

**Run tests:**
```bash
cd backend
pytest tests/ -v
```

## Model Details

### Features (8 Medical Indicators)
1. **Age** (30-80 years)
2. **Resting Blood Pressure** (90-200 mmHg)
3. **Cholesterol** (100-400 mg/dl)
4. **Max Heart Rate** (60-200 bpm)
5. **ST Depression** (0-6, exercise-induced)
6. **Number of Vessels** (0-3, fluoroscopy)
7. **Fasting Blood Sugar** (70-200 mg/dl)
8. **BMI** (18-45)

### Model Performance
- **Training Accuracy**: ~95%
- **Testing Accuracy**: ~90%
- **Cross-Validation**: ~88% (±2%)
- **ROC-AUC**: ~96%
- **F1-Score**: ~90%

### Algorithm Configuration
- **Model**: GradientBoostingClassifier
- **Estimators**: 100
- **Learning Rate**: 0.1
- **Max Depth**: 3 (prevents overfitting)
- **Regularization**: min_samples_split=20, min_samples_leaf=10, subsample=0.8

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /api/v1/predict` - Single patient prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/model/info` - Model information
- `POST /api/v1/model/retrain` - Retrain model
- `GET /api/v1/feature-importances` - Feature importance rankings

## Project Structure
```
002_heart_disease_risk/
├── app.py                      # Standalone ML application
├── heart_disease_model.pkl     # Trained model
├── results.png                 # Visualizations
├── results.txt                 # Metrics report
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   └── ml_service.py      # ML model service
│   ├── tests/
│   │   └── test_main.py       # Comprehensive tests (12 tests)
│   └── requirements.txt       # Dependencies
└── README.md                   # This file
```

## Testing
All 12 tests pass successfully:
- ✅ Health checks and API info
- ✅ Valid patient predictions  
- ✅ High-risk & low-risk scenarios
- ✅ Input validation (age, missing fields)
- ✅ Batch predictions
- ✅ Model info and feature importances
- ✅ Model retraining
- ✅ Prediction consistency

## Improvements Made
1. **Realistic medical features** instead of generic data
2. **Reduced overfitting** through hyperparameter tuning
3. **Enhanced visualizations** (ROC, PR curves)
4. **Full REST API** with production best practices
5. **Comprehensive test suite** (12 tests, 100% pass)
6. **Model persistence** and loading
7. **Cross-validation** for robust metrics
8. **Feature importance analysis**
