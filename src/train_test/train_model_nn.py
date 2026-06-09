import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from features import data, preprocessor

# ------------------ Prepare Data ------------------
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

# Apply preprocessing (fit once on train only)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Fit preprocessor on training and transform all sets
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)

# ------------------ Build Neural Network ------------------

model = Sequential([
    Dense(64, activation="relu", input_dim=X_train.shape[1],
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(32, activation="relu",
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Early stopping
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es],
    verbose=1
)

# ------------------ Validation Results ------------------
y_val_pred = (model.predict(X_val) > 0.5).astype(int)
print("\nValidation Results (Neural Network)\n")
print(classification_report(y_val, y_val_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_val_pred),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Validation Confusion Matrix (Neural Network)")
plt.show()

# ------------------ Test Results ------------------
y_test_pred = (model.predict(X_test) > 0.5).astype(int)
loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Results (Neural Network)\n")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_test_pred),
            annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Test Confusion Matrix (Neural Network)")
plt.show()

# ------------------ Save Model ------------------
model.save("data/processed/heart_model_nn.keras")
print("Model saved as heart_model_nn.keras")
