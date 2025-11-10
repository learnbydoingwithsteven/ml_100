# 🏥 Disease Diagnosis System

## Overview
AI-powered disease diagnosis system using RandomForest classification to analyze patient symptoms and vital signs. This ML application provides comprehensive diagnosis with risk factor analysis and personalized recommendations.

## Use Case
Medical diagnosis assistance based on patient vitals and symptom severity

## Problem Type
Multi-class Classification (5 disease severity levels)

## Features

### Standalone ML Application
- ✅ Synthetic patient data generation
- ✅ RandomForest classifier training
- ✅ Comprehensive evaluation metrics
- ✅ 6-panel visualization dashboard
- ✅ Feature importance analysis
- ✅ Confusion matrix and accuracy metrics
- ✅ Detailed results export

### Full-Stack Application
- ✅ RESTful API with FastAPI backend
- ✅ React TypeScript frontend
- ✅ Real-time disease diagnosis
- ✅ Patient data input forms
- ✅ Visual probability distributions
- ✅ Risk factor identification
- ✅ Treatment recommendations
- ✅ Model information dashboard

## Quick Start

### Standalone ML Script
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Run analysis
python app.py
```

### Full-Stack Application
```bash
# Using Docker Compose (Recommended)
docker-compose up

# Access:
# - Frontend: http://localhost:10001
# - Backend API: http://localhost:9001
# - API Docs: http://localhost:9001/docs
```

### Manual Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 9001

# Frontend (separate terminal)
cd frontend
npm install
npm start
```

## Model Details

### Algorithm
- **Type**: RandomForestClassifier
- **Parameters**: 100 estimators, max_depth=10
- **Training**: 2000 samples, 80/20 train/test split
- **Accuracy**: ~78% on test set

### Features (Patient Metrics)
1. **Body Temperature** (°C): 35-42
2. **Blood Pressure** (mmHg): 60-200
3. **Heart Rate** (BPM): 40-200
4. **Blood Glucose** (mg/dL): 50-400
5. **Cholesterol** (mg/dL): 100-400
6. **BMI**: 10-60
7. **Age**: 0-120
8. **Symptom Severity**: 0-100

### Disease Categories
- **Level 0**: Healthy
- **Level 1**: Mild Condition
- **Level 2**: Moderate Condition
- **Level 3**: Severe Condition
- **Level 4**: Critical Condition

## API Endpoints

### Diagnosis
```bash
POST /api/v1/diagnose
Body: {
  "body_temperature": 37.0,
  "blood_pressure": 120.0,
  "heart_rate": 70.0,
  "glucose_level": 90.0,
  "cholesterol": 180.0,
  "bmi": 22.0,
  "age": 30.0,
  "symptom_severity": 10.0
}
```

### Model Training
```bash
POST /api/v1/train
```

### Model Information
```bash
GET /api/v1/model-info
GET /api/v1/feature-importance
```

### Health Check
```bash
GET /health
GET /
```

## Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v

# Expected: 9 tests passing
```

### Test Coverage
- ✅ Root endpoint
- ✅ Health endpoint
- ✅ Diagnosis with healthy patient
- ✅ Diagnosis with critical patient
- ✅ Invalid data validation
- ✅ Model training
- ✅ Model information
- ✅ Feature importance
- ✅ Edge case handling

## Output Files

### Standalone Script
- **results.png**: 6-panel visualization dashboard
  - Target distribution
  - Confusion matrix
  - Feature importance
  - Feature correlation heatmap
  - Prediction accuracy
  - Feature scatter plot
- **results.txt**: Detailed metrics and classification report

## Technology Stack

### Backend
- FastAPI 0.104.1
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3
- uvicorn 0.24.0

### Frontend
- React 18+ with TypeScript
- Axios for API communication
- Responsive gradient UI design

### Infrastructure
- Docker & Docker Compose
- PostgreSQL (configured)
- Redis (configured)

## Project Structure
```
001_disease_diagnosis/
├── app.py                    # Standalone ML script
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI application
│   ├── tests/
│   │   └── test_main.py     # Comprehensive test suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   └── App.tsx          # React application
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Medical Disclaimer
⚠️ **Important**: This is a demonstration application for educational purposes only. It uses synthetic data and simplified models. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

## Future Enhancements
- [ ] Real medical dataset integration
- [ ] Multi-disease classification
- [ ] Patient history tracking
- [ ] Report generation (PDF)
- [ ] Integration with EHR systems
- [ ] Advanced ML models (Neural Networks)
- [ ] Explainable AI features

## License
MIT License

## Version
2.0.0 - Full-Stack Disease Diagnosis System
