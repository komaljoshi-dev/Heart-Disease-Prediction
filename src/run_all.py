from ocr_utils import run_ocr_pipeline
from simple_predictor import HeartPredictor
import pandas as pd

# Load best trained model using our simple predictor
predictor = HeartPredictor("data/processed/rf_best_model.pkl")

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

    # Use our simple predictor with risk assessment
    result = predictor.predict(numeric_data_mapped)
    
    if result['success']:
        print("\n=== HEART DISEASE PREDICTION RESULT ===")
        print(f"Has Disease: {'YES' if result['has_disease'] else 'NO'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Recommendation: {result['recommendation']}")
        return result['has_disease']
    else:
        print(f"Error: {result['error']}")
        return None

if __name__ == "__main__":
    result = predict_from_ocr("reports/medical_report.png")
    print("Final Prediction:", result)
