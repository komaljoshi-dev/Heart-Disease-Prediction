import os
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold

DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1"]

def get_scoring_for_pipeline(pipeline):
    """
    Return scoring metrics supported by the pipeline.
    """
    scoring = DEFAULT_METRICS.copy()
    clf = pipeline.named_steps["classifier"]

    # Only include roc_auc if classifier supports predict_proba or decision_function
    if hasattr(clf, "predict_proba") or hasattr(clf, "decision_function"):
        scoring.append("roc_auc")
    return scoring

def evaluate_with_cv(pipeline, X, y, cv=10, scoring=None, return_all=False, random_state=42):
    """
    Perform cross-validation and return metrics summary.
    """
    scoring = scoring or get_scoring_for_pipeline(pipeline)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False)
    
    summary = {}
    for metric in scoring:
        arr = results[f"test_{metric}"]
        summary[metric] = {"mean": arr.mean(), "std": arr.std()}
    
    if return_all:
        return summary, results
    return summary

def save_cv_summary(summary_dict, model_name, path="reports/cv_results.csv"):
    """
    Save CV summary to CSV.
    """
    os.makedirs("reports", exist_ok=True)
    rows = []
    for metric, stats in summary_dict.items():
        rows.append({
            "Model": model_name,
            "Metric": metric,
            "Mean": round(stats["mean"], 4),
            "Std": round(stats["std"], 4)
        })
    df = pd.DataFrame(rows)
    
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)
    
    print(f"{model_name} CV results saved to {path}")
    return df

def run_evaluation(pipeline, model_name, X, y, cv=10):
    """
    Wrapper function: perform CV and save results.
    """
    scoring = get_scoring_for_pipeline(pipeline)
    summary = evaluate_with_cv(pipeline, X, y, cv=cv, scoring=scoring)
    df_summary = save_cv_summary(summary, model_name)
    
    print(f"\n{model_name} CV results:\n")
    print(df_summary)
    
    return summary
