import os
import sys

from sklearn.ensemble import RandomForestClassifier
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
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Run evaluation
run_evaluation(pipeline, "random_forest", X, y, cv=10)
