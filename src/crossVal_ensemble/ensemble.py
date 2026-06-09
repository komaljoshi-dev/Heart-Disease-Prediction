import os
import sys
import warnings

from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from evaluation import run_evaluation
from features import data, preprocessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Data
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# Base models
lr = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
svm_rbf = SVC(kernel="rbf", probability=True, random_state=42)
nn = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    solver="adam",
    learning_rate_init=0.001,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42
)

# Pipelines
voting_soft = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm_rbf), ("nn", nn)],
        voting="soft"
    ))
])

voting_hard = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm_rbf), ("nn", nn)],
        voting="hard"
    ))
])

bagging_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", BaggingClassifier(
        estimator=rf, n_estimators=10, random_state=42
    ))
])

stacking = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm_rbf)],
        final_estimator=LogisticRegression(),
        passthrough=True
    ))
])

# -----------------------------
# Run all ensembles
# -----------------------------
pipelines = [
    ("voting_soft", voting_soft),
    ("voting_hard", voting_hard),
    ("bagging_rf", bagging_rf),
    ("stacking", stacking)
]

for name, pipe in pipelines:
    run_evaluation(pipe, name, X, y, cv=10)
