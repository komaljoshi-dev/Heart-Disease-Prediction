import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/interim/cleaned_heart_dataset.csv")

# Continuous numeric columns
numeric_cols = ['age', 'restBP', 'chol', 'max_HR', 'oldpeak']

# Binary / multi-class numeric-coded features
binary_cols = ['sex', 'cp', 'fbs', 'restECG', 'exang', 'slope', 'ca', 'thal']

# Age groups as numeric codes
bins = [0, 30, 40, 50, 60, 100]
labels = [0, 1, 2, 3, 4]  # numeric labels
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False).astype(int)
binary_cols.append('age_group')  # include in numeric-coded categorical features

# StandardScaler for continuous numeric columns
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_cols),
        ('bin', 'passthrough', binary_cols)
    ]
)
X = data.drop(columns=["target_binary", "target_multiclass"])
y = data["target_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test set for evaluation
os.makedirs("data/processed", exist_ok=True)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Features loaded and test set saved at data/processed/")

__all__ = ["data", "preprocessor", "numeric_cols", "binary_cols",
           "X_train", "X_test", "y_train", "y_test"]
