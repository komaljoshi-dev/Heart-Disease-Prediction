from preprocessing.ocr_utils import run_ocr_pipeline
import joblib
import pandas as pd

# Load trained model
model = joblib.load("data/processed/heart_model.pkl")

def predict_from_ocr(image_path):
    numeric_data = run_ocr_pipeline(image_path)
    
    # Map OCR keys to model columns
    mapping = {
        "Age": "age",
        "Sex": "sex",
        "chestPain": "cp",
        "bloodPressure": "restBP",
        "Cholesterol": "chol",
        "bloodSugar": "fbs",
        "ECG": "restECG",
        "maxHeartRate": "max_HR",
        "exerciseAngina": "exang",
        "stDepression": "oldpeak",
        "stSlope": "slope",
        "Thal": "thal"
    }

    numeric_data_mapped = {mapping[k]: v for k, v in numeric_data.items() if k in mapping}

    # Age group numeric
    age = numeric_data_mapped["age"]
    if age < 30:
        numeric_data_mapped["age_group"] = 0
    elif age < 40:
        numeric_data_mapped["age_group"] = 1
    elif age < 50:
        numeric_data_mapped["age_group"] = 2
    elif age < 60:
        numeric_data_mapped["age_group"] = 3
    else:
        numeric_data_mapped["age_group"] = 4

    # Coronary vessels (ca) default 0
    numeric_data_mapped["ca"] = 0

    # Convert to DataFrame
    df = pd.DataFrame([numeric_data_mapped])

    # Predict
    prediction = model.predict(df)[0]

    # Probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        print("\nPrediction (0=no disease/1=disease) : ", prediction)
        print("Probabilities (No disease, Disease) : ", proba)
    else:
        print("Prediction:", prediction)

    return prediction

if __name__ == "__main__":
    result = predict_from_ocr("data/raw/medical_report.png")
    print("Final Prediction:", result)
