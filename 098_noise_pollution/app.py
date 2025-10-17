"""
Noise Pollution Prediction
Use Case #98: Noise Pollution Prediction
Algorithm: RandomForest
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor


print("="*80)
print("NOISE POLLUTION PREDICTION")
print("="*80)


np.random.seed(42)
n_samples = 2000

# Generate features (customize based on use case)
feature_1 = np.random.uniform(0, 100, n_samples)
feature_2 = np.random.uniform(0, 100, n_samples)
feature_3 = np.random.uniform(0, 100, n_samples)
feature_4 = np.random.uniform(0, 100, n_samples)
feature_5 = np.random.uniform(0, 100, n_samples)
feature_6 = np.random.uniform(0, 100, n_samples)
feature_7 = np.random.uniform(0, 100, n_samples)
feature_8 = np.random.uniform(0, 100, n_samples)

# Generate target variable
target = (feature_1 * 0.3 + feature_2 * 0.2 + feature_3 * 0.15 + 
         feature_4 * 0.1 + np.random.normal(0, 10, n_samples))

df = pd.DataFrame({
    'feature_1': feature_1,
    'feature_2': feature_2,
    'feature_3': feature_3,
    'feature_4': feature_4,
    'feature_5': feature_5,
    'feature_6': feature_6,
    'feature_7': feature_7,
    'feature_8': feature_8,
    'target': target
})


print(f"\nDataset Shape: {df.shape}")
print(f"\nTarget Distribution:\n{df['target'].value_counts()}")
print(f"\nDataset Preview:\n{df.head()}")

X = df.drop('target', axis=1)
y = df['target']

# Scale features if needed
if "RandomForest" in ["SVM", "KMeans"]:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "="*80)
print("TRAINING RANDOMFOREST MODEL")
print("="*80)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)
print(f"\nTraining Score: {model.score(X_train, y_train):.4f}")
print(f"Testing Score: {model.score(X_test, y_test):.4f}")

# Visualizations
fig = plt.figure(figsize=(20, 12))

# Plot 1: Target Distribution
ax1 = plt.subplot(2, 3, 1)
if "regression" == "regression":
    ax1.hist(y, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Target Value')
    ax1.set_ylabel('Frequency')
else:
    target_counts = df['target'].value_counts()
    ax1.bar(range(len(target_counts)), target_counts.values, color=plt.cm.Set3(range(len(target_counts))))
    ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    for i, v in enumerate(target_counts.values):
        ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
ax1.grid(alpha=0.3)

# Plot 2: Predictions vs Actual
ax2 = plt.subplot(2, 3, 2)
if "regression" == "regression":
    ax2.scatter(y_test, y_pred, alpha=0.5, s=30)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.grid(alpha=0.3)
else:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

# Plot 3: Feature Importance (if available)
ax3 = plt.subplot(2, 3, 3)
if hasattr(model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
    ax3.barh(importance['feature'], importance['importance'], color=colors)
    ax3.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Importance')
    ax3.invert_yaxis()
else:
    ax3.text(0.5, 0.5, 'Feature Importance\nNot Available', 
            ha='center', va='center', fontsize=12)
    ax3.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Feature Correlation
ax4 = plt.subplot(2, 3, 4)
corr = X.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax4, square=True)
ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# Plot 5: Residuals or Error Distribution
ax5 = plt.subplot(2, 3, 5)
if "regression" == "regression":
    residuals = y_test - y_pred
    ax5.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax5.axhline(y=0, color='r', linestyle='--', lw=2)
    ax5.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Predicted Values')
    ax5.set_ylabel('Residuals')
    ax5.grid(alpha=0.3)
else:
    errors = (y_test != y_pred).astype(int)
    ax5.bar(['Correct', 'Incorrect'], [sum(errors==0), sum(errors==1)], 
           color=['green', 'red'], alpha=0.7)
    ax5.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Count')
    for i, v in enumerate([sum(errors==0), sum(errors==1)]):
        ax5.text(i, v + 5, str(v), ha='center', fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')

# Plot 6: Feature Scatter
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(df['feature_1'], df['feature_2'], c=df['target'], 
                     cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax6.set_title('Feature 1 vs Feature 2', fontsize=14, fontweight='bold')
ax6.set_xlabel('Feature 1')
ax6.set_ylabel('Feature 2')
plt.colorbar(scatter, ax=ax6, label='Target')
ax6.grid(alpha=0.3)

plt.tight_layout()
output_dir = 'c:/Users/wjbea/Downloads/learnbydoingwithsteven/ml_100/098_noise_pollution'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'results.png'")

# Save results
with open(f'{output_dir}/results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("NOISE POLLUTION PREDICTION - DETAILED RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset Size: {len(df)} samples\n")
    f.write(f"Training Set: {len(X_train)} samples\n")
    f.write(f"Testing Set: {len(X_test)} samples\n\n")
    f.write(f"Training Score: {model.score(X_train, y_train):.4f}\n")
    f.write(f"Testing Score: {model.score(X_test, y_test):.4f}\n\n")
    if "regression" == "regression":
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        f.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}\n")
        f.write(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\n")
        f.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}\n")
        f.write(f"R² Score: {r2_score(y_test, y_pred):.4f}\n")
    else:
        from sklearn.metrics import classification_report
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

print("✓ Detailed results saved as 'results.txt'")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
plt.show()
