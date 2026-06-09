# src/evaluate_test/metrics_utils.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar


def compute_basic_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    return metrics


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    McNemar's test for paired classification results:
    Returns test stat and p-value.
    """
    a = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    d = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    table = [[a, b], [c, d]]
    # exact=False uses chi-square approx (with continuity correction if correction=True)
    res = mcnemar(table, exact=False, correction=True)
    return {"table": table, "statistic": res.statistic, "pvalue": res.pvalue}


def bootstrap_auc_compare(
    y_true, prob1, prob2, n_bootstraps=1000, seed=42
):
    """
    Bootstrap test for difference in AUC between two models.
    Returns mean difference, 95% CI, and two-sided p-value.
    """
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y_true)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc1 = roc_auc_score(y_true[idx], prob1[idx])
        auc2 = roc_auc_score(y_true[idx], prob2[idx])
        diffs.append(auc1 - auc2)
    diffs = np.array(diffs)
    mean_diff = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: proportion of bootstrap diffs <= 0 or >=0
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"mean_diff": mean_diff, "ci": (ci_low, ci_high), "pvalue": p_value}
