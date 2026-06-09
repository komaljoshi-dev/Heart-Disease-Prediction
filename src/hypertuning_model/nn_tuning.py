import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from features import data, preprocessor
from evaluation import run_evaluation
import joblib

# Data
X = data.drop(columns=["target_binary","target_multiclass"])
y = data["target_binary"]

#Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(max_iter=500, random_state=42))
])

#Hyperparameter grid
param_grid = {
    "classifier__hidden_layer_sizes": [(50,), (100,), (100,50)],
    "classifier__learning_rate_init": [0.001, 0.01, 0.1],
    "classifier__alpha": [0.0001, 0.001, 0.01]  # L2 regularization
}

#Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

#best model
best_model = grid_search.best_estimator_
print("Best NN params:", grid_search.best_params_)

#CV evaluation
run_evaluation(best_model, "nn_optimized", X, y, cv=10)

#Save model
os.makedirs("data/processed", exist_ok=True)
joblib.dump(best_model, "data/processed/nn_best_model.pkl")
print("Neural Network model saved to data/processed/nn_best_model.pkl")
