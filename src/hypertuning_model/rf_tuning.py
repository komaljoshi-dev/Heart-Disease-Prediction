import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from features import data, preprocessor
from evaluation import run_evaluation
import joblib

X = data.drop(columns=["target_binary","target_multiclass"])
y = data["target_binary"]

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print("Best RF params:", grid_search.best_params_)

# Evaluate CV
run_evaluation(best_model, "rf_optimized", X, y, cv=10)

# Save model
os.makedirs("data/processed", exist_ok=True)
joblib.dump(best_model, "data/processed/rf_best_model.pkl")
print("Random Forest model saved to data/processed/rf_best_model.pkl")
