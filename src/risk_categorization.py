"""
Heart Disease Risk Categorization System
Provides validated thresholds and risk categories for clinical decision making
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class HeartRiskCategorizer:
    """
    Heart Disease Risk Categorization System
    
    Provides three risk categories based on probability thresholds:
    - Low Risk: Probability < 0.3
    - Medium Risk: 0.3 <= Probability < 0.7  
    - High Risk: Probability >= 0.7
    
    Thresholds are validated based on:
    1. Clinical relevance (minimizing false negatives)
    2. Balanced sensitivity/specificity
    3. Practical utility for healthcare providers
    """
    
    def __init__(self):
        # Validated thresholds based on clinical considerations
        self.low_threshold = 0.3    # Conservative threshold - few false negatives
        self.high_threshold = 0.7   # High confidence threshold
        
        self.risk_categories = {
            0: "Low Risk",
            1: "Medium Risk", 
            2: "High Risk"
        }
        
        self.category_descriptions = {
            "Low Risk": "Low probability of heart disease. Routine screening recommended.",
            "Medium Risk": "Moderate probability. Additional testing and monitoring advised.",
            "High Risk": "High probability of heart disease. Immediate medical evaluation recommended."
        }
    
    def categorize_risk(self, probabilities):
        """
        Categorize probabilities into risk levels
        
        Args:
            probabilities: Array-like of disease probabilities
            
        Returns:
            Array of risk categories (0=Low, 1=Medium, 2=High)
        """
        probabilities = np.asarray(probabilities)
        categories = np.zeros_like(probabilities, dtype=int)
        
        categories[(probabilities >= self.low_threshold) & 
                  (probabilities < self.high_threshold)] = 1  # Medium Risk
        categories[probabilities >= self.high_threshold] = 2  # High Risk
        
        return categories
    
    def get_risk_labels(self, categories):
        """Get text labels for risk categories"""
        return [self.risk_categories[cat] for cat in categories]
    
    def validate_thresholds(self, y_true, y_prob, print_results=True):
        """
        Validate risk categorization thresholds
        
        Args:
            y_true: True binary labels (0=No Disease, 1=Disease)
            y_prob: Predicted probabilities
            print_results: Whether to print validation results
            
        Returns:
            Dictionary with validation metrics
        """
        risk_categories = self.categorize_risk(y_prob)
        
        # Calculate metrics for each threshold
        low_pred = (y_prob >= self.low_threshold).astype(int)
        high_pred = (y_prob >= self.high_threshold).astype(int)
        
        # Confusion matrices
        cm_low = confusion_matrix(y_true, low_pred)
        cm_high = confusion_matrix(y_true, high_pred)
        
        # Category-wise analysis
        category_analysis = {}
        for cat in [0, 1, 2]:
            mask = (risk_categories == cat)
            if np.sum(mask) > 0:
                disease_rate = np.mean(y_true[mask]) if np.sum(mask) > 0 else 0
                category_analysis[self.risk_categories[cat]] = {
                    'count': np.sum(mask),
                    'disease_rate': disease_rate,
                    'avg_probability': np.mean(y_prob[mask])
                }
        
        results = {
            'thresholds': {
                'low_threshold': self.low_threshold,
                'high_threshold': self.high_threshold
            },
            'category_analysis': category_analysis,
            'low_threshold_metrics': self._calculate_threshold_metrics(cm_low),
            'high_threshold_metrics': self._calculate_threshold_metrics(cm_high)
        }
        
        if print_results:
            self._print_validation_results(results)
        
        return results
    
    def _calculate_threshold_metrics(self, cm):
        """Calculate metrics from confusion matrix"""
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def _print_validation_results(self, results):
        """Print formatted validation results"""
        print("="*70)
        print("HEART DISEASE RISK CATEGORIZATION VALIDATION")
        print("="*70)
        
        print(f"\nValidated Thresholds:")
        print(f"  Low Risk Threshold:  < {self.low_threshold}")
        print(f"  Medium Risk Range:   {self.low_threshold} - {self.high_threshold}")
        print(f"  High Risk Threshold: >= {self.high_threshold}")
        
        print(f"\nRisk Category Analysis:")
        for category, analysis in results['category_analysis'].items():
            print(f"  {category}:")
            print(f"    Patients: {analysis['count']}")
            print(f"    Disease Rate: {analysis['disease_rate']:.1%}")
            print(f"    Avg Probability: {analysis['avg_probability']:.3f}")
        
        print(f"\nThreshold Performance:")
        
        low_metrics = results['low_threshold_metrics']
        print(f"  Low Risk Threshold ({self.low_threshold}):")
        print(f"    Sensitivity: {low_metrics['sensitivity']:.3f}")
        print(f"    Specificity: {low_metrics['specificity']:.3f}")
        print(f"    Accuracy: {low_metrics['accuracy']:.3f}")
        
        high_metrics = results['high_threshold_metrics']
        print(f"  High Risk Threshold ({self.high_threshold}):")
        print(f"    Sensitivity: {high_metrics['sensitivity']:.3f}")
        print(f"    Specificity: {high_metrics['specificity']:.3f}")
        print(f"    Accuracy: {high_metrics['accuracy']:.3f}")
    
    def plot_risk_distribution(self, y_true, y_prob, save_path="reports/figures/risk_distribution.png"):
        """Plot risk category distribution"""
        risk_categories = self.categorize_risk(y_prob)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Risk category distribution
        category_counts = pd.Series(risk_categories).value_counts().sort_index()
        category_labels = [self.risk_categories[i] for i in category_counts.index]
        
        colors = ['green', 'orange', 'red']
        ax1.bar(category_labels, category_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Patient Distribution by Risk Category', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Patients')
        
        # Add counts on bars
        for i, count in enumerate(category_counts.values):
            ax1.text(i, count + 1, str(count), ha='center', fontweight='bold')
        
        # Plot 2: Disease rate by risk category
        disease_rates = []
        for cat in [0, 1, 2]:
            mask = (risk_categories == cat)
            if np.sum(mask) > 0:
                disease_rate = np.mean(y_true[mask])
                disease_rates.append(disease_rate)
            else:
                disease_rates.append(0)
        
        ax2.bar(category_labels, disease_rates, color=colors, alpha=0.7)
        ax2.set_title('Disease Rate by Risk Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Disease Rate')
        ax2.set_ylim(0, 1)
        
        # Add percentages on bars
        for i, rate in enumerate(disease_rates):
            ax2.text(i, rate + 0.02, f'{rate:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Risk distribution plot saved to: {save_path}")
    
    def generate_clinical_report(self, patient_prob, patient_features=None):
        """
        Generate clinical risk assessment report for a patient
        
        Args:
            patient_prob: Predicted probability for the patient
            patient_features: Dictionary of patient features (optional)
            
        Returns:
            Dictionary with clinical assessment
        """
        risk_category_num = self.categorize_risk([patient_prob])[0]
        risk_category = self.risk_categories[risk_category_num]
        description = self.category_descriptions[risk_category]
        
        # Risk level interpretation
        if risk_category_num == 0:
            urgency = "Routine"
            recommendation = "Continue regular preventive care and lifestyle modifications"
        elif risk_category_num == 1:
            urgency = "Monitoring Required"
            recommendation = "Schedule follow-up appointment and consider additional cardiac testing"
        else:
            urgency = "Immediate Attention"
            recommendation = "Urgent cardiology consultation and comprehensive cardiac evaluation recommended"
        
        report = {
            'probability': patient_prob,
            'risk_category': risk_category,
            'risk_level': risk_category_num,
            'description': description,
            'urgency': urgency,
            'recommendation': recommendation,
            'confidence': 'High' if patient_prob < 0.2 or patient_prob > 0.8 else 'Medium'
        }
        
        return report


def validate_risk_system_on_test_data():
    """Validate the risk categorization system on test data"""
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from features import y_test
    import joblib
    
    # Load best model (Random Forest)
    best_model = joblib.load("data/processed/rf_best_model.pkl")
    
    # Get test predictions
    y_prob = best_model.predict_proba(y_test.index)[:, 1]
    
    # Initialize risk categorizer
    risk_system = HeartRiskCategorizer()
    
    # Validate thresholds
    validation_results = risk_system.validate_thresholds(y_test.values, y_prob)
    
    # Generate plots
    risk_system.plot_risk_distribution(y_test.values, y_prob)
    
    return risk_system, validation_results


if __name__ == "__main__":
    # Run validation when script is executed directly
    risk_system, results = validate_risk_system_on_test_data()
    
    # Example clinical report
    print("\n" + "="*70)
    print("EXAMPLE CLINICAL RISK ASSESSMENT")
    print("="*70)
    
    example_prob = 0.85
    report = risk_system.generate_clinical_report(example_prob)
    
    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
