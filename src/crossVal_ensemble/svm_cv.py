import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from evaluation import run_evaluation
from features import data, preprocessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# Pipelines
pipeline_linear = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear", probability=True, random_state=42))
])

pipeline_rbf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
])

# Run evaluation
run_evaluation(pipeline_linear, "svm_linear", X, y, cv=10)
run_evaluation(pipeline_rbf, "svm_rbf", X, y, cv=10)
