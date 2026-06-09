"""
Complete Heart Disease Prediction Pipeline
Ready for UI integration with preprocessing, prediction, and risk categorization
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from risk_categorization import HeartRiskCategorizer
except ImportError:
    print("Warning: risk_categorization module not found. Risk categorization will be limited.")
    HeartRiskCategorizer = None

# Import functions for overall model performance
from src.final_performance_report import load_evaluation_results, generate_executive_summary

class HeartDiseasePredictionPipeline:
    """
    Complete end-to-end prediction pipeline for heart disease risk assessment.
    
    Features:
    - Input validation and preprocessing
    - Model prediction with calibrated probabilities
    - Risk categorization (Low/Medium/High)
    - Clinical report generation
    - UI-ready JSON output
    """
    
    def __init__(self, model_path="data/processed/calibrated/random_forest_calibrated.pkl"):
        """
        Initialize the prediction pipeline
        
        Args:
            model_path: Path to the trained and calibrated model
        """
        self.model_path = model_path
        self.model = None
        self.risk_categorizer = None
        self.overall_model_accuracy = None # New attribute for overall accuracy
        self.feature_names = [
            'age', 'sex', 'cp', 'restBP', 'chol', 'fbs', 
            'restECG', 'max_HR', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal', 'age_group'
        ]
        
        # Feature descriptions for UI
        self.feature_descriptions = {
            'age': 'Age (years)',
            'sex': 'Sex (0: Female, 1: Male)',
            'cp': 'Chest Pain Type (1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic)',
            'restBP': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Cholesterol Level (mg/dl)',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)',
            'restECG': 'Resting ECG Results (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)',
            'max_HR': 'Maximum Heart Rate Achieved',
            'exang': 'Exercise Induced Angina (0: No, 1: Yes)',
            'oldpeak': 'ST Depression Induced by Exercise',
            'slope': 'Slope of Peak Exercise ST Segment (1: Upsloping, 2: Flat, 3: Downsloping)',
            'ca': 'Number of Major Vessels (0-3)',
            'thal': 'Thalassemia (3: Normal, 6: Fixed Defect, 7: Reversible Defect)',
            'age_group': 'Age Group (Auto-calculated from age)'
        }
        
        # Valid ranges for input validation
        self.feature_ranges = {
            'age': (20, 100),
            'sex': (0, 1),
            'cp': (1, 4),
            'restBP': (80, 220),
            'chol': (100, 600),
            'fbs': (0, 1),
            'restECG': (0, 2),
            'max_HR': (60, 220),
            'exang': (0, 1),
            'oldpeak': (0.0, 8.0),
            'slope': (1, 3),
            'ca': (0, 3),
            'thal': (3, 7),
            'age_group': (0, 4)
        }
        
        self.load_model()
        self.initialize_risk_categorizer()
        self._load_overall_model_accuracy()
    
    def _load_overall_model_accuracy(self):
        """Load the overall model accuracy from the performance report."""
        try:
            results_data = load_evaluation_results()
            if 'test_evaluation' in results_data:
                summary = generate_executive_summary(results_data)
                self.overall_model_accuracy = summary['best_model']['accuracy']
            else:
                print("Warning: No test evaluation results found for overall model accuracy.")
                self.overall_model_accuracy = 0.0
        except Exception as e:
            print(f"Error loading overall model accuracy: {e}")
            self.overall_model_accuracy = 0.0
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        else:
            # Fallback to original model if calibrated not available
            fallback_path = "data/processed/rf_best_model.pkl"
            if os.path.exists(fallback_path):
                self.model = joblib.load(fallback_path)
                print(f"Fallback model loaded from {fallback_path}")
            else:
                raise FileNotFoundError(f"No model found at {self.model_path} or {fallback_path}")
    
    def initialize_risk_categorizer(self):
        """Initialize the risk categorization system"""
        if HeartRiskCategorizer:
            self.risk_categorizer = HeartRiskCategorizer()
        else:
            print("Warning: Risk categorization system not available")
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        else:
            # Fallback to original model if calibrated not available
            fallback_path = "data/processed/rf_best_model.pkl"
            if os.path.exists(fallback_path):
                self.model = joblib.load(fallback_path)
                print(f"Fallback model loaded from {fallback_path}")
            else:
                raise FileNotFoundError(f"No model found at {self.model_path} or {fallback_path}")
    
    def initialize_risk_categorizer(self):
        """Initialize the risk categorization system"""
        if HeartRiskCategorizer:
            self.risk_categorizer = HeartRiskCategorizer()
        else:
            print("Warning: Risk categorization system not available")
    
    def calculate_age_group(self, age):
        """Calculate age group from age"""
        if age < 30:
            return 0
        elif age < 40:
            return 1
        elif age < 50:
            return 2
        elif age < 60:
            return 3
        else:
            return 4
    
    def validate_input(self, patient_data):
        """
        Validate input data
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        # Check required features
        required_features = [f for f in self.feature_names if f != 'age_group']
        for feature in required_features:
            if feature not in patient_data:
                errors.append(f"Missing required field: {feature}")
        
        if errors:
            return False, errors
        
        # Validate ranges
        for feature, value in patient_data.items():
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                try:
                    numeric_value = float(value)
                    if not (min_val <= numeric_value <= max_val):
                        errors.append(f"{feature} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"{feature} must be a valid number")
        
        return len(errors) == 0, errors
    
    def preprocess_input(self, patient_data):
        """
        Preprocess input data for prediction
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            numpy array ready for prediction
        """
        # Create a copy to avoid modifying original
        data = patient_data.copy()
        
        # Calculate age group if not provided
        if 'age_group' not in data and 'age' in data:
            data['age_group'] = self.calculate_age_group(float(data['age']))
        
        # Create feature vector in correct order
        feature_vector = []
        for feature in self.feature_names:
            if feature in data:
                feature_vector.append(float(data[feature]))
            else:
                raise ValueError(f"Missing feature: {feature}")
        
        # Convert to DataFrame to match training format
        df = pd.DataFrame([feature_vector], columns=self.feature_names)
        
        return df
    
    def predict(self, patient_data):
        """
        Make prediction for a patient
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            Dictionary with prediction results
        """
        # Validate input
        is_valid, errors = self.validate_input(patient_data)
        if not is_valid:
            return {
                'success': False,
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(patient_data)
            
            # Make prediction
            prediction_binary = self.model.predict(processed_data)[0]
            prediction_proba = self.model.predict_proba(processed_data)[0, 1]
            
            # Generate risk assessment
            risk_assessment = self.generate_risk_assessment(
                prediction_proba, patient_data
            )
            
            # Create comprehensive result
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'input_data': patient_data,
                'prediction': {
                    'has_heart_disease': bool(prediction_binary),
                    'probability': float(prediction_proba),
                    'confidence': 'High' if prediction_proba < 0.2 or prediction_proba > 0.8 else 'Medium'
                },
                'risk_assessment': risk_assessment,
                'model_info': {
                    'model_type': str(type(self.model).__name__),
                    'model_path': self.model_path
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_risk_assessment(self, probability, patient_data):
        """
        Generate clinical risk assessment
        
        Args:
            probability: Predicted probability
            patient_data: Original patient data
            
        Returns:
            Dictionary with risk assessment
        """
        if self.risk_categorizer:
            return self.risk_categorizer.generate_clinical_report(probability, patient_data)
        else:
            # Basic risk categorization if advanced system not available
            if probability < 0.3:
                risk_level = "Low Risk"
                urgency = "Routine"
                recommendation = "Continue regular preventive care"
            elif probability < 0.7:
                risk_level = "Medium Risk"
                urgency = "Monitoring Required"
                recommendation = "Consider additional cardiac testing"
            else:
                risk_level = "High Risk"
                urgency = "Immediate Attention"
                recommendation = "Urgent cardiology consultation recommended"
            
            return {
                'probability': probability,
                'risk_category': risk_level,
                'urgency': urgency,
                'recommendation': recommendation,
                'confidence': 'High' if probability < 0.2 or probability > 0.8 else 'Medium'
            }
    
    def batch_predict(self, patients_data):
        """
        Make predictions for multiple patients
        
        Args:
            patients_data: List of patient data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for i, patient_data in enumerate(patients_data):
            result = self.predict(patient_data)
            result['patient_id'] = i
            results.append(result)
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance from the model (if available)
        
        Returns:
            Dictionary with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance = np.abs(self.model.coef_[0])
        else:
            # For pipeline or other models, try to access the classifier
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    importance = classifier.feature_importances_
                elif hasattr(classifier, 'coef_'):
                    importance = np.abs(classifier.coef_[0])
                else:
                    return None
            else:
                return None
        
        # Normalize to percentages
        importance_pct = (importance / importance.sum()) * 100
        
        # Create dictionary with feature names
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = {
                'importance': float(importance_pct[i]),
                'description': self.feature_descriptions.get(feature, feature)
            }
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), 
                          key=lambda x: x[1]['importance'], reverse=True))
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': str(type(self.model).__name__),
            'model_path': self.model_path,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'feature_descriptions': self.feature_descriptions,
            'feature_ranges': self.feature_ranges
        }
        
        # Add feature importance if available
        feature_importance = self.get_feature_importance()
        if feature_importance:
            info['feature_importance'] = feature_importance
        
        return info
    
    def save_prediction_to_file(self, prediction_result, filename=None):
        """
        Save prediction result to JSON file
        
        Args:
            prediction_result: Result from predict() method
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp}.json"
        
        output_dir = "reports/predictions"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(prediction_result, f, indent=2, default=str)
        
        return filepath


def create_example_patient_data():
    """Create example patient data for testing"""
    return {
        'age': 65,
        'sex': 1,  # Male
        'cp': 2,   # Atypical angina
        'restBP': 140,
        'chol': 250,
        'fbs': 0,  # No
        'restECG': 1,
        'max_HR': 150,
        'exang': 1,  # Yes
        'oldpeak': 2.3,
        'slope': 2,
        'ca': 2,
        'thal': 3  # Reversible defect
    }


def demo_prediction_pipeline():
    """Demonstrate the prediction pipeline"""
    print("="*70)
    print("HEART DISEASE PREDICTION PIPELINE DEMO")
    print("="*70)
    
    # Initialize pipeline
    pipeline = HeartDiseasePredictionPipeline()
    
    # Create example patient
    patient_data = create_example_patient_data()
    
    print("Example Patient Data:")
    for key, value in patient_data.items():
        description = pipeline.feature_descriptions.get(key, key)
        print(f"  {description}: {value}")
    
    # Make prediction
    result = pipeline.predict(patient_data)
    
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Get model info
    model_info = pipeline.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model Type: {model_info['model_type']}")
    print(f"  Features: {model_info['feature_count']}")
    
    # Save result
    saved_path = pipeline.save_prediction_to_file(result)
    print(f"\nPrediction saved to: {saved_path}")
    
    return pipeline, result


if __name__ == "__main__":
    pipeline, result = demo_prediction_pipeline()