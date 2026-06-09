# src/evaluate_test/model_comparison.py
import itertools
import os
import sys

import joblib
import pandas as pd

from metrics_utils import (
    mcnemar_test, bootstrap_auc_compare
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pred_path = "reports/test_predictions.pkl"
if not os.path.exists(pred_path):
    raise FileNotFoundError(
        pred_path + " not found. Run evaluate_test_set.py first."
    )

preds = joblib.load(pred_path)  # dict: {model_name: {"y_pred":...,
                                 # "y_prob":...}}

models = list(preds.keys())
rows = []

# Pairwise McNemar on test predictions
for m1, m2 in itertools.combinations(models, 2):
    y_pred1 = preds[m1]["y_pred"]
    y_pred2 = preds[m2]["y_pred"]
    # load y_test from features
    from features import y_test
    mrt = mcnemar_test(y_test.values, y_pred1, y_pred2)
    row = {
        "model_1": m1, "model_2": m2,
        "mcnemar_stat": mrt["statistic"], "mcnemar_p": mrt["pvalue"]
    }
    # If both models have probabilities, compare AUC by bootstrap
    prob1 = preds[m1]["y_prob"]
    prob2 = preds[m2]["y_prob"]
    if prob1 is not None and prob2 is not None:
        boot = bootstrap_auc_compare(
            y_test.values, prob1, prob2, n_bootstraps=2000
        )
        row.update({
            "auc_diff": boot["mean_diff"],
            "auc_diff_ci_low": boot["ci"][0],
            "auc_diff_ci_high": boot["ci"][1],
            "auc_boot_p": boot["pvalue"]
        })
    rows.append(row)

df = pd.DataFrame(rows)
os.makedirs("reports", exist_ok=True)
df.to_csv("reports/model_pairwise_stats.csv", index=False)
print(" Model comparison saved to reports/model_pairwise_stats.csv")
