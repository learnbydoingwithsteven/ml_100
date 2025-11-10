"""
Heart Disease Risk Prediction ML Service
Handles model training, prediction, and management
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_realistic_heart_disease_data(n_samples=2000, random_state=42):
    """
    Generate synthetic but realistic heart disease data
    Based on common heart disease risk factors
    """
    np.random.seed(random_state)
    
    # Age (30-80 years) - older age increases risk
    age = np.random.normal(55, 12, n_samples).clip(30, 80)
    
    # Resting Blood Pressure (90-200 mmHg) - higher BP increases risk
    resting_bp = np.random.normal(130, 20, n_samples).clip(90, 200)
    
    # Cholesterol (100-400 mg/dl) - higher cholesterol increases risk
    cholesterol = np.random.normal(240, 50, n_samples).clip(100, 400)
    
    # Max Heart Rate (60-200 bpm) - lower max HR can indicate issues
    max_heart_rate = np.random.normal(150, 22, n_samples).clip(60, 200)
    
    # ST Depression (0-6) - exercise-induced ST depression
    st_depression = np.random.exponential(1.2, n_samples).clip(0, 6)
    
    # Number of major vessels (0-3) - colored by fluoroscopy
    num_vessels = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.25, 0.15, 0.1])
    
    # Fasting Blood Sugar (70-200 mg/dl) - >120 is concerning
    fasting_bs = np.random.normal(110, 30, n_samples).clip(70, 200)
    
    # BMI (18-45) - obesity increases risk
    bmi = np.random.normal(27, 5, n_samples).clip(18, 45)
    
    # Generate target based on realistic medical relationships
    risk_score = (
        (age - 30) * 0.8 +                    # Age factor
        (resting_bp - 120) * 0.5 +            # BP factor
        (cholesterol - 200) * 0.3 +           # Cholesterol factor
        (150 - max_heart_rate) * 0.4 +        # Heart rate factor
        st_depression * 15 +                  # ST depression (strong indicator)
        num_vessels * 25 +                    # Vessel blockage (strong indicator)
        (fasting_bs - 100) * 0.3 +            # Blood sugar factor
        (bmi - 25) * 2                        # BMI factor
    )
    
    # Add some noise to make it more realistic
    risk_score += np.random.normal(0, 15, n_samples)
    
    # Convert to binary classification (0: low risk, 1: high risk)
    target = (risk_score > np.percentile(risk_score, 50)).astype(int)
    
    df = pd.DataFrame({
        'age': age.round(0),
        'resting_bp': resting_bp.round(0),
        'cholesterol': cholesterol.round(0),
        'max_heart_rate': max_heart_rate.round(0),
        'st_depression': st_depression.round(2),
        'num_vessels': num_vessels,
        'fasting_bs': fasting_bs.round(0),
        'bmi': bmi.round(1),
        'target': target
    })
    
    return df


class HeartDiseaseModel:
    """Heart Disease Risk Prediction Model"""
    
    FEATURE_NAMES = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 
                     'st_depression', 'num_vessels', 'fasting_bs', 'bmi']
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the model"""
        self.model = None
        self.feature_importances = None
        self.training_metrics = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No pre-trained model found. Training new model...")
            self.train()
    
    def train(self, n_samples: int = 2000) -> Dict:
        """Train the model on synthetic data"""
        logger.info(f"Generating {n_samples} training samples...")
        df = generate_realistic_heart_disease_data(n_samples=n_samples)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Training Gradient Boosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        self.training_metrics = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_accuracy': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'n_samples': n_samples,
            'n_features': len(self.FEATURE_NAMES)
        }
        
        # Store feature importances
        self.feature_importances = dict(zip(
            self.FEATURE_NAMES,
            self.model.feature_importances_.tolist()
        ))
        
        logger.info(f"Model trained successfully. Test accuracy: {test_acc:.4f}")
        return self.training_metrics
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Make a prediction for a single patient
        
        Args:
            features: Dictionary with patient features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Validate features
        for feature in self.FEATURE_NAMES:
            if feature not in features:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[self.FEATURE_NAMES]
        
        # Make prediction
        pred_array = self.model.predict(X)
        prob_array = self.model.predict_proba(X)
        
        prediction = int(pred_array[0])
        probability = float(prob_array[0][1])
        
        return {
            'risk_level': 'high' if prediction == 1 else 'low',
            'risk_score': prediction,
            'probability': probability,
            'confidence': max(probability, 1 - probability),
            'features': features
        }
    
    def predict_batch(self, patients: List[Dict[str, float]]) -> List[Dict]:
        """Make predictions for multiple patients"""
        return [self.predict(patient) for patient in patients]
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_importances': self.feature_importances,
            'training_metrics': self.training_metrics,
            'feature_names': self.FEATURE_NAMES
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        model_data = joblib.load(path)
        
        # Handle both bare model and dictionary formats
        if isinstance(model_data, dict):
            # Dictionary format from ml_service
            self.model = model_data['model']
            self.feature_importances = model_data.get('feature_importances')
            self.training_metrics = model_data.get('training_metrics', {})
        else:
            # Bare model format from standalone app.py
            self.model = model_data
            self.feature_importances = None
            self.training_metrics = {}
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances"""
        if self.feature_importances is None and self.model is not None and hasattr(self.model, 'feature_importances_'):
            # Compute if not cached
            self.feature_importances = dict(zip(
                self.FEATURE_NAMES,
                self.model.feature_importances_.tolist()
            ))
        return self.feature_importances or {}
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'trained': self.model is not None,
            'feature_names': self.FEATURE_NAMES,
            'feature_importances': self.get_feature_importances(),
            'metrics': self.training_metrics or {}
        }


# Global model instance
_model_instance = None


def get_model() -> HeartDiseaseModel:
    """Get or create the global model instance"""
    global _model_instance
    if _model_instance is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'heart_disease_model.pkl')
        _model_instance = HeartDiseaseModel(model_path if os.path.exists(model_path) else None)
    return _model_instance
