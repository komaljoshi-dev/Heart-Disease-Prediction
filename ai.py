import pandas as pd
import glob

# Column names from heart-disease.names
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# List of all processed dataset files
files = [
    "data/processed.cleveland.data",
    "data/processed.hungarian.data",
    "data/processed.switzerland.data",
    "data/processed.va.data"
]

# Load and combine
df_list = []
for file in files:
    temp = pd.read_csv(file, header=None)
    temp.columns = columns
    df_list.append(temp)

# Merge all datasets into one DataFrame
df = pd.concat(df_list, ignore_index=True)

# Replace missing values marked as '?' with NaN
df = df.replace('?', pd.NA)

# Convert numeric columns to proper types
for col in ["ca", "thal"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values (or you can handle them differently)
df = df.dropna()

# Save merged dataset to CSV
df.to_csv("heart_disease_merged.csv", index=False)

print(" Merged dataset created: heart_disease_merged.csv")
print("Shape:", df.shape)
print(df.head())