import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluation import run_evaluation
from features import data, preprocessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        solver="liblinear", max_iter=1000, random_state=42
    ))
])

# Run evaluation
run_evaluation(pipeline, "logistic_regression", X, y, cv=10)
