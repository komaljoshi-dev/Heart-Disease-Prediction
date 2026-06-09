import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from features import data, preprocessor

# Prepare data
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# Train / validation / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)

# ---------- 1. Linear Kernel ----------
pipeline_linear = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear", probability=True, random_state=42))
])

pipeline_linear.fit(X_train, y_train)

# Validation
y_val_pred_linear = pipeline_linear.predict(X_val)
print("\nValidation Results (SVM - Linear Kernel)\n")
print(classification_report(y_val, y_val_pred_linear))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_val_pred_linear),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Validation Confusion Matrix (SVM - Linear)")
plt.show()

# Test
y_test_pred_linear = pipeline_linear.predict(X_test)
print("\nTest Results (SVM - Linear Kernel)\n")
print("Accuracy:", accuracy_score(y_test, y_test_pred_linear))
print(classification_report(y_test, y_test_pred_linear))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred_linear),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Test Confusion Matrix (SVM - Linear)")
plt.show()

# Save
joblib.dump(pipeline_linear, "data/processed/heart_model_svm_linear.pkl")
print("Model saved as heart_model_svm_linear.pkl")


# ---------- 2. RBF Kernel ----------
pipeline_rbf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
])

pipeline_rbf.fit(X_train, y_train)

# Validation
y_val_pred_rbf = pipeline_rbf.predict(X_val)
print("\nValidation Results (SVM - RBF Kernel)\n")
print(classification_report(y_val, y_val_pred_rbf))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_val_pred_rbf),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Validation Confusion Matrix (SVM - RBF)")
plt.show()

# Test
y_test_pred_rbf = pipeline_rbf.predict(X_test)
print("\nTest Results (SVM - RBF Kernel)\n")
print("Accuracy:", accuracy_score(y_test, y_test_pred_rbf))
print(classification_report(y_test, y_test_pred_rbf))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred_rbf),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Test Confusion Matrix (SVM - RBF)")
plt.show()

# Save
joblib.dump(pipeline_rbf, "data/processed/heart_model_svm_rbf.pkl")
print("Model saved as heart_model_svm_rbf.pkl")
