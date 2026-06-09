import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from features import data, preprocessor
from evaluation import run_evaluation
import joblib

# Data
X = data.drop(columns=["target_binary","target_multiclass"])
y = data["target_binary"]

#  Feature Selection
lr = LogisticRegression(solver="liblinear", penalty="l1", max_iter=1000, random_state=42)
rfe = RFE(estimator=lr, n_features_to_select=8)

#  Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", rfe),
    ("classifier", lr)
])

# CV evaluation 
run_evaluation(pipeline, "lr_lasso_rfe", X, y, cv=10)

# 🔧 FIX: Fit the pipeline on full dataset (CRITICAL STEP!)
print("\n🔄 Fitting pipeline on full dataset...")
pipeline.fit(X, y)
print("✅ Pipeline fitted successfully!")

# Save selected features (get them from the fitted pipeline)
fitted_rfe = pipeline.named_steps["feature_selector"]
selected_features = X.columns[fitted_rfe.support_].tolist()
print("\n📋 Selected features:", selected_features)
os.makedirs("reports", exist_ok=True)
pd.DataFrame(selected_features, columns=["Feature"]).to_csv("reports/selected_features.csv", index=False)

# Save fitted pipeline
os.makedirs("data/processed", exist_ok=True)
joblib.dump(pipeline, "data/processed/lr_lasso_rfe_pipeline.pkl")
print("\n✅ Fitted Logistic Regression RFE pipeline saved to data/processed/lr_lasso_rfe_pipeline.pkl")
