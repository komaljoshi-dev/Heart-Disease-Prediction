# src/evaluate_test/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os

def plot_and_save_confusion_matrix(cm, model_name, outdir="reports/figures"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{model_name}_confusion_matrix.png")
    plt.close()

def plot_and_save_roc(y_true, y_prob, model_name, outdir="reports/figures"):
    os.makedirs(outdir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{outdir}/{model_name}_roc.png")
    plt.close()
    return roc_auc

def plot_and_save_pr(y_true, y_prob, model_name, outdir="reports/figures"):
    os.makedirs(outdir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{outdir}/{model_name}_pr.png")
    plt.close()
    return pr_auc
