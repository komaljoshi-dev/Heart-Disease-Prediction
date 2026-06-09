import os
import sys

from sklearn.neural_network import MLPClassifier
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
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=500, random_state=42
    ))
])

# Run evaluation
run_evaluation(pipeline, "neural_network", X, y, cv=10)
