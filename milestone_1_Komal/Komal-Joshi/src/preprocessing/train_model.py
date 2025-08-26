import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from features import data,preprocessor,numeric_cols,binary_cols

X = data.drop(columns=["target_binary","target_multiclass"])
y = data["target_binary"]

# Train / validation / test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nTraining data shape :",X_train.shape)
print("Validation data shape :",X_val.shape)
print("Test data shape :",X_test.shape,)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

#evaluate on validation
y_val_pred = pipeline.predict(X_val)
print("\nBaseline Validation Results\n")
print(classification_report(y_val, y_val_pred))
# Validation Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_val_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#Final Test evaluation
y_test_pred = pipeline.predict(X_test)
print("\nTest Results\n")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Test Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred), 
            annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

X_train.to_csv("data/processed/X_train.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)

X_val.to_csv("data/processed/X_val.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)

X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

#Save model
joblib.dump(pipeline, "data/processed/heart_model.pkl")
print("Model saved as heart_model.pkl")
