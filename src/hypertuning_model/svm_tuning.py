import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from features import data, preprocessor
from evaluation import run_evaluation
import joblib
from scipy.stats import expon, uniform

X = data.drop(columns=["target_binary","target_multiclass"])
y = data["target_binary"]

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, random_state=42))
])

param_dist = {
    "classifier__C": uniform(0.1, 10),      # continuous range from 0.1 to 10
    "classifier__gamma": expon(scale=0.1),  # exponential distribution
    "classifier__kernel": ["linear", "rbf"] # try both kernels
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,           # number of random combinations
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X, y)

best_model = random_search.best_estimator_
print("Best SVM params:", random_search.best_params_)

run_evaluation(best_model, "svm_optimized", X, y)

os.makedirs("data/processed", exist_ok=True)
joblib.dump(best_model, "data/processed/svm_best_model.pkl")
print("SVM model saved to data/processed/svm_best_model.pkl")
