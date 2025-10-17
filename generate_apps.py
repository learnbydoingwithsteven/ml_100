"""
Master Generator for 100 ML Applications
Creates complete, independent apps with full functionality
"""

import os
import json

def get_app_template(app_info):
    """Generate complete Python app code based on app configuration"""
    
    app_id = app_info['id']
    name = app_info['name']
    title = app_info['title']
    algorithm = app_info['algorithm']
    app_type = app_info['type']
    n_classes = app_info['classes']
    
    # Determine imports based on algorithm
    imports = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
"""
    
    # Algorithm-specific imports
    if algorithm == "RandomForest":
        if app_type == "regression":
            imports += "from sklearn.ensemble import RandomForestRegressor\n"
            model_line = "model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)"
        else:
            imports += "from sklearn.ensemble import RandomForestClassifier\n"
            model_line = "model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)"
    elif algorithm == "GradientBoosting":
        if app_type == "regression":
            imports += "from sklearn.ensemble import GradientBoostingRegressor\n"
            model_line = "model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)"
        else:
            imports += "from sklearn.ensemble import GradientBoostingClassifier\n"
            model_line = "model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)"
    elif algorithm == "LogisticRegression":
        imports += "from sklearn.linear_model import LogisticRegression\n"
        model_line = "model = LogisticRegression(max_iter=1000, random_state=42)"
    elif algorithm == "LinearRegression":
        imports += "from sklearn.linear_model import LinearRegression\n"
        model_line = "model = LinearRegression()"
    elif algorithm == "SVM":
        imports += "from sklearn.svm import SVC, SVR\nfrom sklearn.preprocessing import StandardScaler\n"
        if app_type == "regression":
            model_line = "model = SVR(kernel='rbf')"
        else:
            model_line = "model = SVC(kernel='rbf', probability=True, random_state=42)"
    elif algorithm == "KMeans":
        imports += "from sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\n"
        model_line = f"model = KMeans(n_clusters={n_classes}, random_state=42, n_init=10)"
    elif algorithm == "XGBoost":
        imports += "try:\n    from xgboost import XGBClassifier, XGBRegressor\nexcept:\n    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier\n    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor\n"
        if app_type == "regression":
            model_line = "model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)"
        else:
            model_line = "model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)"
    elif algorithm == "NaiveBayes":
        imports += "from sklearn.naive_bayes import GaussianNB\n"
        model_line = "model = GaussianNB()"
    elif algorithm == "IsolationForest":
        imports += "from sklearn.ensemble import IsolationForest\n"
        model_line = "model = IsolationForest(contamination=0.1, random_state=42)"
    else:
        # Default to RandomForest
        if app_type == "regression":
            imports += "from sklearn.ensemble import RandomForestRegressor\n"
            model_line = "model = RandomForestRegressor(n_estimators=100, random_state=42)"
        else:
            imports += "from sklearn.ensemble import RandomForestClassifier\n"
            model_line = "model = RandomForestClassifier(n_estimators=100, random_state=42)"
    
    # Generate data creation code
    data_gen = f"""
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
"""
    
    if app_type == "regression":
        data_gen += """target = (feature_1 * 0.3 + feature_2 * 0.2 + feature_3 * 0.15 + 
         feature_4 * 0.1 + np.random.normal(0, 10, n_samples))
"""
    elif app_type == "binary":
        data_gen += """score = (feature_1 * 0.3 + feature_2 * 0.2 + feature_3 * 0.15)
target = (score > np.percentile(score, 60)).astype(int)
"""
    elif app_type == "classification" and n_classes > 0:
        data_gen += f"""score = (feature_1 * 0.3 + feature_2 * 0.2 + feature_3 * 0.15)
percentiles = np.linspace(0, 100, {n_classes + 1})
target = np.digitize(score, np.percentile(score, percentiles[1:-1]))
"""
    else:
        data_gen += """score = (feature_1 * 0.3 + feature_2 * 0.2 + feature_3 * 0.15)
target = (score > np.percentile(score, 60)).astype(int)
"""
    
    data_gen += """
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
"""
    
    # Complete app code
    code = f'''"""
{title}
Use Case #{app_id}: {title}
Algorithm: {algorithm}
"""

{imports}

print("="*80)
print("{title.upper()}")
print("="*80)

{data_gen}

print(f"\\nDataset Shape: {{df.shape}}")
print(f"\\nTarget Distribution:\\n{{df['target'].value_counts()}}")
print(f"\\nDataset Preview:\\n{{df.head()}}")

X = df.drop('target', axis=1)
y = df['target']

# Scale features if needed
if "{algorithm}" in ["SVM", "KMeans"]:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\\n" + "="*80)
print("TRAINING {algorithm.upper()} MODEL")
print("="*80)

{model_line}
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)
print(f"\\nTraining Score: {{model.score(X_train, y_train):.4f}}")
print(f"Testing Score: {{model.score(X_test, y_test):.4f}}")

# Visualizations
fig = plt.figure(figsize=(20, 12))

# Plot 1: Target Distribution
ax1 = plt.subplot(2, 3, 1)
if "{app_type}" == "regression":
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
if "{app_type}" == "regression":
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
    importance = pd.DataFrame({{
        'feature': X.columns,
        'importance': model.feature_importances_
    }}).sort_values('importance', ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
    ax3.barh(importance['feature'], importance['importance'], color=colors)
    ax3.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Importance')
    ax3.invert_yaxis()
else:
    ax3.text(0.5, 0.5, 'Feature Importance\\nNot Available', 
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
if "{app_type}" == "regression":
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
output_dir = 'c:/Users/wjbea/Downloads/learnbydoingwithsteven/ml_100/{app_id:03d}_{name}'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{{output_dir}}/results.png', dpi=300, bbox_inches='tight')
print("\\n✓ Visualization saved as 'results.png'")

# Save results
with open(f'{{output_dir}}/results.txt', 'w') as f:
    f.write("="*80 + "\\n")
    f.write("{title.upper()} - DETAILED RESULTS\\n")
    f.write("="*80 + "\\n\\n")
    f.write(f"Dataset Size: {{len(df)}} samples\\n")
    f.write(f"Training Set: {{len(X_train)}} samples\\n")
    f.write(f"Testing Set: {{len(X_test)}} samples\\n\\n")
    f.write(f"Training Score: {{model.score(X_train, y_train):.4f}}\\n")
    f.write(f"Testing Score: {{model.score(X_test, y_test):.4f}}\\n\\n")
    if "{app_type}" == "regression":
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        f.write(f"Mean Squared Error: {{mean_squared_error(y_test, y_pred):.4f}}\\n")
        f.write(f"Root Mean Squared Error: {{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}}\\n")
        f.write(f"Mean Absolute Error: {{mean_absolute_error(y_test, y_pred):.4f}}\\n")
        f.write(f"R² Score: {{r2_score(y_test, y_pred):.4f}}\\n")
    else:
        from sklearn.metrics import classification_report
        f.write("Classification Report:\\n")
        f.write(classification_report(y_test, y_pred))

print("✓ Detailed results saved as 'results.txt'")
print("\\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
plt.show()
'''
    
    return code

def create_readme(app_info):
    """Generate README for each app"""
    app_id = app_info['id']
    name = app_info['name']
    title = app_info['title']
    algorithm = app_info['algorithm']
    app_type = app_info['type']
    
    readme = f"""# {title}

## Use Case #{app_id}
{title} using {algorithm} algorithm

## Problem Type
{app_type.capitalize()}

## Features
- Synthetic data generation
- {algorithm} model training
- Comprehensive evaluation metrics
- 6 visualization plots
- Detailed results export

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
```bash
python app.py
```

## Output Files
- **results.png**: 6-panel visualization dashboard
- **results.txt**: Detailed numerical results and metrics

## Model Details
- **Algorithm**: {algorithm}
- **Features**: 8 numerical features
- **Evaluation**: Train/test split with comprehensive metrics
"""
    
    return readme

def generate_all_apps():
    """Generate all 100 ML applications"""
    
    # Load catalog
    with open('c:/Users/wjbea/Downloads/learnbydoingwithsteven/ml_100/app_catalog.json', 'r') as f:
        apps = json.load(f)
    
    print(f"Generating {len(apps)} ML applications...")
    print("="*80)
    
    for app in apps:
        app_id = app['id']
        name = app['name']
        title = app['title']
        
        # Create directory
        dir_path = f"c:/Users/wjbea/Downloads/learnbydoingwithsteven/ml_100/{app_id:03d}_{name}"
        os.makedirs(dir_path, exist_ok=True)
        
        # Generate app.py
        app_code = get_app_template(app)
        with open(f"{dir_path}/app.py", 'w', encoding='utf-8') as f:
            f.write(app_code)
        
        # Generate README.md
        readme_content = create_readme(app)
        with open(f"{dir_path}/README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ {app_id:3d}. {title:40s} [{app['algorithm']}]")
    
    print("="*80)
    print(f"\\n✓ Successfully generated all {len(apps)} applications!")
    print("\\nEach application includes:")
    print("  - app.py (complete implementation)")
    print("  - README.md (documentation)")
    print("  - Auto-generated results.png and results.txt on execution")

if __name__ == "__main__":
    generate_all_apps()
