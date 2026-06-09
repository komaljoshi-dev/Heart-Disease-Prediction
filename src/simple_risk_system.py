"""
Simple Heart Disease Risk Categorization
"""
import numpy as np

class RiskCategorizer:
    def __init__(self):
        self.low_threshold = 0.3
        self.high_threshold = 0.7
    
    def categorize(self, probability):
        """Return risk category: Low/Medium/High"""
        if probability < self.low_threshold:
            return "Low Risk"
        elif probability < self.high_threshold:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def get_recommendation(self, probability):
        """Get clinical recommendation"""
        category = self.categorize(probability)
        
        if category == "Low Risk":
            return "Routine preventive care recommended"
        elif category == "Medium Risk":
            return "Additional testing and monitoring advised"
        else:
            return "Urgent cardiology consultation recommended"

# Usage example
if __name__ == "__main__":
    risk = RiskCategorizer()
    
    # Test probabilities
    test_probs = [0.15, 0.45, 0.85]
    
    for prob in test_probs:
        category = risk.categorize(prob)
        recommendation = risk.get_recommendation(prob)
        print(f"Probability: {prob:.1%} -> {category}")
        print(f"Recommendation: {recommendation}\n")
