import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from features import data, preprocessor
from evaluation import run_evaluation
import joblib

# ----------------- Data -----------------
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# ----------------- Pipeline -----------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(solver="liblinear", max_iter=2000, random_state=42))
])

# ----------------- Hyperparameter Grid -----------------
param_grid = {
    "classifier__penalty": ["l1", "l2"],    # regularization type
    "classifier__C": [0.01, 0.1, 1, 10, 100]  # inverse of regularization strength
}

# ----------------- Grid Search -----------------
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

# ----------------- Best Model -----------------
best_model = grid_search.best_estimator_
print("Best Logistic Regression params:", grid_search.best_params_)

# Evaluate on CV again for consistency
run_evaluation(best_model, "logistic_regression_optimized", X, y)

# Save the model
os.makedirs("data/processed", exist_ok=True)
joblib.dump(best_model, "data/processed/logistic_best_model.pkl")
print("Logistic Regression model saved to data/processed/logistic_best_model.pkl")
