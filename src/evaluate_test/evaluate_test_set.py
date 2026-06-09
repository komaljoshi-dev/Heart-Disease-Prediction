# src/evaluate_test/evaluate_test_set.py
import os
import sys

import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from features import X_test, X_train, preprocessor, y_test
from metrics_utils import compute_basic_metrics, specificity_score
from plot_utils import (
    plot_and_save_confusion_matrix,
    plot_and_save_roc,
    plot_and_save_pr
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def safe_predict(model, X_data, model_name="Unknown"):
    """
    Safely predict using a model, handling both pipeline and standalone models.

    Args:
        model: Loaded sklearn model (could be pipeline or standalone)
        X_data: Raw input data (pandas DataFrame)
        model_name: Name of the model for logging

    Returns:
        tuple: (y_pred, y_prob) where y_prob is None if not available
    """
    print(f"  Evaluating {model_name}...")
    # Check if model is a pipeline
    if isinstance(model, Pipeline):
        print(f"    → Pipeline detected: {[name for name, _ in model.steps]}")
        X_input = X_data  # Pipeline handles preprocessing internally
    else:
        print(f"    → Standalone model detected: {type(model).__name__}")
        # For standalone models, we need to fit and transform the data
        try:
            # Check if preprocessor is fitted by looking for fitted attributes
            is_fitted = (hasattr(preprocessor, 'transformers_') and
                         preprocessor.transformers_ is not None)
            if not is_fitted:
                print("    → Fitting preprocessor on training data...")
                preprocessor.fit(X_train)
            X_input = preprocessor.transform(X_data)
            print(f"    → Data preprocessed: {X_input.shape}")
        except Exception as e:
            print(f"    → Preprocessing failed: {e}")
            raise
    # Make predictions
    try:
        y_pred = model.predict(X_input)
        print(f"    → Predictions made: {len(y_pred)} samples")
    except NotFittedError as e:
        print(f"    → Model not fitted error: {e}")
        raise
    except Exception as e:
        print(f"    → Prediction error: {e}")
        raise
    # Get probabilities if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_input)[:, 1]
            print(f"    → Probabilities extracted: {len(y_prob)} samples")
        except Exception as e:
            print(f"    → Probability extraction failed: {e}")
            y_prob = None
    else:
        print("    → No predict_proba method available")
    return y_pred, y_prob


# Configure models to evaluate (ensure these files exist)
models = {
    "Logistic Regression (RFE)": "data/processed/lr_lasso_rfe_pipeline.pkl",
    "Logistic regression optimized": "data/processed/logistic_best_model.pkl",
    "Random Forest Optimized": "data/processed/rf_best_model.pkl",
    "SVM Optimized": "data/processed/svm_best_model.pkl",
    "Neural Network Optimized": "data/processed/nn_best_model.pkl"
}

os.makedirs("reports/figures", exist_ok=True)
all_results = []
predictions = {}   # store preds & probs for later comparisons

for name, path in models.items():
    if not os.path.exists(path):
        print(f"Model not found: {path}. Skipping {name}.")
        continue
    print(f"\nLoading model: {name}")
    model = joblib.load(path)
    
    # Use safe prediction that handles both pipelines and standalone models
    try:
        y_pred, y_prob = safe_predict(model, X_test, name)
    except Exception as e:
        print(f"Failed to evaluate {name}: {e}")
        continue

    metrics = compute_basic_metrics(y_test, y_pred, y_prob)
    metrics["specificity"] = specificity_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plot_and_save_confusion_matrix(cm, name)
    if y_prob is not None:
        roc_auc = plot_and_save_roc(y_test, y_prob, name)
        pr_auc = plot_and_save_pr(y_test, y_prob, name)
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    all_results.append({"Model": name, **metrics})
    predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}

# Save metrics
df_results = pd.DataFrame(all_results)
os.makedirs("reports", exist_ok=True)
df_results.to_csv("reports/test_set_evaluation.csv", index=False)
# Save predictions for model_comparison
joblib.dump(predictions, "reports/test_predictions.pkl")
print("Test evaluation finished. Metrics: reports/test_set_evaluation.csv")
