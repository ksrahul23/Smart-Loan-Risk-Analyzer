import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Add current directory to path
sys.path.append(os.getcwd())
from src.preprocess import get_preprocessor, clean_dependents

# 1. Load Data
df = pd.read_csv("data/loan_data.csv")

# 2. Pre-cleaning (Dependents conversion)
df = clean_dependents(df)

# 3. Model Preparation
# Features and Target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create Pipeline
preprocessor = get_preprocessor()
rf_model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# 5. Hyperparameter Tuning
param_dist = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

print("Starting hyperparameter tuning...")
random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=5, 
    random_state=42, 
    n_jobs=-1,
    scoring='accuracy'
)

random_search.fit(X_train, y_train)
best_pipeline = random_search.best_estimator_

# 6. Evaluation
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")

# 7. Save Model and Metrics
joblib.dump(best_pipeline, "models/loan_model.pkl")

# Save metrics for the UI
metrics = {
    "accuracy": round(acc, 2),
    "auc": round(auc, 2),
    "confusion_matrix": cm.tolist(),
    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Model and metrics saved successfully!")