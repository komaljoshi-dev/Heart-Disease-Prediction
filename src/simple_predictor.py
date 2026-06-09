"""
Simple Heart Disease Predictor - UI Ready
"""
import pandas as pd
import joblib
import os
from simple_risk_system import RiskCategorizer

class HeartPredictor:
    def __init__(self, model_path="data/processed/rf_best_model.pkl"):
        # Load model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.risk_system = RiskCategorizer()
        
        # Expected features in order
        self.features = [
            'age', 'sex', 'cp', 'restBP', 'chol', 'fbs',
            'restECG', 'max_HR', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal', 'age_group'
        ]
    
    def calculate_age_group(self, age):
        """Convert age to age group"""
        if age < 30: return 0
        elif age < 40: return 1
        elif age < 50: return 2  
        elif age < 60: return 3
        else: return 4
    
    def predict(self, patient_data):
        """
        Make prediction for patient
        
        Args:
            patient_data: dict with patient features
        
        Returns:
            dict with prediction results
        """
        try:
            # Add age group if missing
            if 'age_group' not in patient_data:
                patient_data['age_group'] = self.calculate_age_group(patient_data['age'])
            
            # Create feature vector
            feature_values = [patient_data[feature] for feature in self.features]
            df = pd.DataFrame([feature_values], columns=self.features)
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0, 1]
            
            # Get risk assessment
            risk_category = self.risk_system.categorize(probability)
            recommendation = self.risk_system.get_recommendation(probability)
            
            return {
                'success': True,
                'has_disease': bool(prediction),
                'probability': float(probability),
                'risk_category': risk_category,
                'recommendation': recommendation
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    predictor = HeartPredictor()
    
    # Example patient data
    patient = {
        'age': 65, 'sex': 1, 'cp': 2, 'restBP': 140, 'chol': 250,
        'fbs': 0, 'restECG': 1, 'max_HR': 150, 'exang': 1,
        'oldpeak': 2.3, 'slope': 2, 'ca': 2, 'thal': 3
    }
    
    result = predictor.predict(patient)
    print("Prediction Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
