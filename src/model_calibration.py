"""
Model Calibration for Heart Disease Prediction
Applies calibration methods to ensure reliable probability outputs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import os

class ModelCalibrator:
    """
    Model calibration system that applies and validates calibration methods
    to ensure reliable probability outputs for clinical decision making.
    """
    
    def __init__(self):
        self.calibration_methods = ['platt', 'isotonic']
        self.calibrated_models = {}
        self.calibration_curves_data = {}
    
    def calibrate_model(self, model, X_train, y_train, X_val, y_val, method='platt'):
        """
        Calibrate a model using specified method
        
        Args:
            model: Trained model to calibrate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            method: Calibration method ('platt' or 'isotonic')
            
        Returns:
            Calibrated model
        """
        print(f"Calibrating model using {method} scaling...")
        
        if method == 'platt':
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        else:  # isotonic
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        
        # Fit calibrator on validation set
        calibrated_model.fit(X_val, y_val)
        
        return calibrated_model
    
    def evaluate_calibration(self, model, X_test, y_test, model_name="Model", n_bins=10):
        """
        Evaluate calibration quality of a model
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name for plotting
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        # Get predicted probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=n_bins, normalize=False
        )
        
        # Calculate Brier score
        brier_score = np.mean((y_prob - y_test) ** 2)
        
        # Calculate reliability (calibration error)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        total_samples = len(y_test)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_test[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Store calibration curve data for plotting
        self.calibration_curves_data[model_name] = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'y_prob': y_prob,
            'y_test': y_test
        }
        
        metrics = {
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
        
        print(f"{model_name} Calibration Metrics:")
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  Calibration Error: {calibration_error:.4f}")
        
        return metrics
    
    def plot_calibration_curve(self, models_data=None, save_path="reports/figures/calibration_curves.png"):
        """
        Plot calibration curves for multiple models
        
        Args:
            models_data: Dictionary of model calibration data (uses stored data if None)
            save_path: Path to save the plot
        """
        if models_data is None:
            models_data = self.calibration_curves_data
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, data) in enumerate(models_data.items()):
            color = colors[i % len(colors)]
            
            plt.plot(data['mean_predicted_value'], data['fraction_of_positives'], 
                    marker='o', linewidth=2, color=color, 
                    label=f'{model_name} (Brier: {data["brier_score"]:.3f})')
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Calibration curves saved to: {save_path}")
    
    def plot_reliability_diagram(self, model_name, save_path=None):
        """
        Plot reliability diagram for a specific model
        
        Args:
            model_name: Name of model to plot
            save_path: Path to save plot
        """
        if model_name not in self.calibration_curves_data:
            print(f"No calibration data found for {model_name}")
            return
        
        data = self.calibration_curves_data[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reliability diagram
        ax1.plot(data['mean_predicted_value'], data['fraction_of_positives'], 
                marker='o', linewidth=2, markersize=8, label=model_name)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title(f'Reliability Diagram - {model_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predicted probabilities
        ax2.hist(data['y_prob'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of Predicted Probabilities - {model_name}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability diagram saved to: {save_path}")
        
        plt.show()
    
    def calibrate_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Calibrate all available models and compare performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of calibrated models and metrics
        """
        # Load available models
        model_paths = {
            "Random Forest": "data/processed/rf_best_model.pkl",
            "Neural Network": "data/processed/nn_best_model.pkl",
            "SVM": "data/processed/svm_best_model.pkl",
            "Logistic Regression": "data/processed/logistic_best_model.pkl"
        }
        
        calibration_results = {}
        
        print("="*70)
        print("MODEL CALIBRATION ANALYSIS")
        print("="*70)
        
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            print(f"\nProcessing {model_name}...")
            
            # Load original model
            original_model = joblib.load(model_path)
            
            # Evaluate original model calibration
            original_metrics = self.evaluate_calibration(
                original_model, X_test, y_test, f"{model_name} (Original)"
            )
            
            # Calibrate using both methods
            calibrated_models = {}
            
            for method in self.calibration_methods:
                calibrated_model = self.calibrate_model(
                    original_model, X_train, y_train, X_val, y_val, method
                )
                
                # Evaluate calibrated model
                calibrated_metrics = self.evaluate_calibration(
                    calibrated_model, X_test, y_test, f"{model_name} ({method.title()})"
                )
                
                calibrated_models[method] = {
                    'model': calibrated_model,
                    'metrics': calibrated_metrics
                }
            
            calibration_results[model_name] = {
                'original': {'model': original_model, 'metrics': original_metrics},
                'calibrated': calibrated_models
            }
        
        return calibration_results
    
    def save_best_calibrated_models(self, calibration_results, output_dir="data/processed/calibrated/"):
        """
        Save the best calibrated models based on Brier score
        
        Args:
            calibration_results: Results from calibrate_all_models
            output_dir: Directory to save calibrated models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving best calibrated models to {output_dir}...")
        
        best_models_summary = []
        
        for model_name, results in calibration_results.items():
            original_brier = results['original']['metrics']['brier_score']
            
            # Find best calibration method
            best_method = None
            best_brier = original_brier
            best_model = results['original']['model']
            
            for method, cal_results in results['calibrated'].items():
                brier = cal_results['metrics']['brier_score']
                if brier < best_brier:
                    best_brier = brier
                    best_method = method
                    best_model = cal_results['model']
            
            # Save best model
            model_filename = f"{model_name.lower().replace(' ', '_')}_calibrated.pkl"
            model_path = os.path.join(output_dir, model_filename)
            joblib.dump(best_model, model_path)
            
            improvement = ((original_brier - best_brier) / original_brier) * 100 if original_brier > 0 else 0
            
            best_models_summary.append({
                'Model': model_name,
                'Best Method': best_method if best_method else 'Original',
                'Original Brier': original_brier,
                'Best Brier': best_brier,
                'Improvement': f"{improvement:.1f}%",
                'Saved Path': model_path
            })
            
            print(f"  {model_name}: {best_method if best_method else 'Original'} "
                  f"(Brier: {best_brier:.4f})")
        
        # Save summary
        summary_df = pd.DataFrame(best_models_summary)
        summary_path = os.path.join(output_dir, "calibration_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nCalibration summary saved to: {summary_path}")
        return summary_df


def run_model_calibration():
    """Run complete model calibration analysis"""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from features import X_train, X_test, y_train, y_test
    from sklearn.model_selection import train_test_split
    
    # Create validation set from training data
    X_train_cal, X_val, y_train_cal, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize calibrator
    calibrator = ModelCalibrator()
    
    # Run calibration analysis
    results = calibrator.calibrate_all_models(
        X_train_cal, y_train_cal, X_val, y_val, X_test, y_test
    )
    
    # Plot calibration curves
    calibrator.plot_calibration_curve()
    
    # Save best models
    summary = calibrator.save_best_calibrated_models(results)
    
    return calibrator, results, summary


if __name__ == "__main__":
    calibrator, results, summary = run_model_calibration()
    
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))
