import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from features import data, preprocessor

# Prepare data
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        solver="liblinear", max_iter=1000, random_state=42
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Validation evaluation
y_val_pred = pipeline.predict(X_val)
print("\nValidation Results (Logistic Regression)\n")
print(classification_report(y_val, y_val_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_val_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Validation Confusion Matrix")
plt.show()

# Test evaluation
y_test_pred = pipeline.predict(X_test)
print("\nTest Results (Logistic Regression)\n")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Test Confusion Matrix")
plt.show()

# Save model
joblib.dump(pipeline, "data/processed/heart_model_logistic.pkl")
print("Model saved as heart_model_logistic.pkl")
