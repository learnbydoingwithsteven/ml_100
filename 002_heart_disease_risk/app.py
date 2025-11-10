"""
Heart Disease Risk Prediction
Use Case #2: Heart Disease Risk Assessment
Algorithm: GradientBoosting
Features: Realistic medical indicators for heart disease prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
import joblib


print("="*80)
print("HEART DISEASE RISK PREDICTION")
print("="*80)


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


df = generate_realistic_heart_disease_data(n_samples=2000, random_state=42)


print(f"\nDataset Shape: {df.shape}")
print(f"\nFeature Statistics:")
print(df.describe().round(2))
print(f"\nTarget Distribution:\n{df['target'].value_counts()}")
print(f"\n{'Class 0 (Low Risk):':<25} {df['target'].value_counts()[0]} ({df['target'].value_counts()[0]/len(df)*100:.1f}%)")
print(f"{'Class 1 (High Risk):':<25} {df['target'].value_counts()[1]} ({df['target'].value_counts()[1]/len(df)*100:.1f}%)")

X = df.drop('target', axis=1)
y = df['target']

# Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*80)
print("TRAINING GRADIENT BOOSTING MODEL")
print("="*80)

# Optimized hyperparameters to prevent overfitting
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,              # Reduced from 5 to prevent overfitting
    min_samples_split=20,     # Increased to prevent overfitting
    min_samples_leaf=10,      # Added to prevent overfitting
    subsample=0.8,            # Added for regularization
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Cross-validation score
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)
print(f"\nTraining Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Testing Accuracy:  {model.score(X_test, y_test):.4f}")
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Visualizations
fig = plt.figure(figsize=(20, 12))

# Plot 1: Target Distribution
ax1 = plt.subplot(2, 3, 1)
target_counts = df['target'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.bar(['Low Risk', 'High Risk'], target_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate(target_counts.values):
    percentage = v/len(df)*100
    ax1.text(i, v + 20, f'{v}\n({percentage:.1f}%)', ha='center', fontweight='bold', fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar_kws={'label': 'Count'},
            xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# Plot 3: Feature Importance
ax3 = plt.subplot(2, 3, 3)
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance)))
bars = ax3.barh(importance['feature'], importance['importance'], color=colors, edgecolor='black', linewidth=1)
ax3.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax3.set_xlabel('Importance Score')
ax3.invert_yaxis()
ax3.grid(alpha=0.3, axis='x')

# Plot 4: ROC Curve
ax4 = plt.subplot(2, 3, 4)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax4.legend(loc="lower right")
ax4.grid(alpha=0.3)

# Plot 5: Precision-Recall Curve
ax5 = plt.subplot(2, 3, 5)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
ax5.plot(recall_vals, precision_vals, color='blue', lw=2, label='PR curve')
ax5.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', label='Baseline')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax5.legend(loc="upper right")
ax5.grid(alpha=0.3)

# Plot 6: Age vs Cholesterol by Risk
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(df['age'], df['cholesterol'], c=df['target'], 
                     cmap='RdYlGn_r', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax6.set_title('Age vs Cholesterol by Risk Level', fontsize=14, fontweight='bold')
ax6.set_xlabel('Age (years)')
ax6.set_ylabel('Cholesterol (mg/dl)')
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Risk Level')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'])
ax6.grid(alpha=0.3)

plt.tight_layout()

# Use current directory for output
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'results.png'), dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'results.png'")

# Save detailed results
results_path = os.path.join(output_dir, 'results.txt')
with open(results_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("HEART DISEASE RISK PREDICTION - DETAILED RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Samples: {len(df)}\n")
    f.write(f"Training Set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)\n")
    f.write(f"Testing Set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)\n")
    f.write(f"Features: {list(X.columns)}\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Training Accuracy: {model.score(X_train, y_train):.4f}\n")
    f.write(f"Testing Accuracy: {model.score(X_test, y_test):.4f}\n")
    f.write(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})\n\n")
    
    f.write("CLASSIFICATION METRICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred):.4f}\n")
    f.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}\n\n")
    
    f.write("DETAILED CLASSIFICATION REPORT\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    f.write("\n" + "="*80 + "\n")
    f.write("FEATURE IMPORTANCE RANKINGS\n")
    f.write("="*80 + "\n")
    for idx, row in importance.iterrows():
        f.write(f"{row['feature']:<20}: {row['importance']:.4f}\n")

print(f"✓ Detailed results saved as 'results.txt'")

# Save the trained model
model_path = os.path.join(output_dir, 'heart_disease_model.pkl')
joblib.dump(model, model_path)
print(f"✓ Model saved as 'heart_disease_model.pkl'")

# Save sample data for testing
sample_data = X_test.head(5).copy()
sample_data['actual_risk'] = y_test.head(5).values
sample_data['predicted_risk'] = y_pred[:5]
sample_data['risk_probability'] = y_pred_proba[:5]
sample_path = os.path.join(output_dir, 'sample_predictions.csv')
sample_data.to_csv(sample_path, index=False)
print(f"✓ Sample predictions saved as 'sample_predictions.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print(f"  - results.png (visualizations)")
print(f"  - results.txt (detailed metrics)")
print(f"  - heart_disease_model.pkl (trained model)")
print(f"  - sample_predictions.csv (sample predictions)")
plt.show()
