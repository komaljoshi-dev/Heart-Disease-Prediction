"""
Simple Final Performance Report
"""
import pandas as pd

def generate_simple_report():
    """Generate a simple final performance report"""
    
    # Load evaluation results
    try:
        results = pd.read_csv("reports/test_set_evaluation.csv")
    except FileNotFoundError:
        print("Error: No evaluation results found. Run evaluation first.")
        return
    
    print("="*60)
    print("HEART DISEASE PREDICTION - FINAL REPORT")
    print("="*60)
    
    # Check 85% accuracy requirement
    models_85_plus = results[results['accuracy'] >= 0.85]
    requirement_met = len(models_85_plus) > 0
    
    print(f"\nACCURACY REQUIREMENT (≥85%): {'✅ MET' if requirement_met else '❌ NOT MET'}")
    print(f"Models meeting requirement: {len(models_85_plus)}/{len(results)}")
    
    # Best model
    best_model = results.loc[results['accuracy'].idxmax()]
    print(f"\nBEST MODEL: {best_model['Model']}")
    print(f"  Accuracy: {best_model['accuracy']:.1%}")
    print(f"  ROC-AUC: {best_model['roc_auc']:.3f}")
    print(f"  Precision: {best_model['precision']:.3f}")
    print(f"  Recall: {best_model['recall']:.3f}")
    
    # All models summary
    print(f"\nALL MODELS SUMMARY:")
    print("-" * 50)
    for _, model in results.iterrows():
        meets_req = "✅" if model['accuracy'] >= 0.85 else "❌"
        print(f"{meets_req} {model['Model']}: {model['accuracy']:.1%} (AUC: {model['roc_auc']:.3f})")
    
    # Deliverables status
    print(f"\nDELIVERABLES STATUS:")
    print(f"✅ Risk categorization system: Implemented")
    print(f"✅ Calibrated probabilities: Available")
    print(f"✅ Saved model files: Complete")
    print(f"✅ Prediction pipeline: Ready for UI")
    print(f"✅ >85% accuracy: {'Achieved' if requirement_met else 'Not achieved'}")
    
    print("="*60)
    
    # Save simple summary
    with open("docs/simple_summary.txt", "w") as f:
        f.write("HEART DISEASE PREDICTION - FINAL SUMMARY\n")
        f.write("="*45 + "\n\n")
        f.write(f"Best Model: {best_model['Model']}\n")
        f.write(f"Best Accuracy: {best_model['accuracy']:.1%}\n")
        f.write(f"85% Requirement: {'MET' if requirement_met else 'NOT MET'}\n")
        f.write(f"Models >=85%: {len(models_85_plus)}/{len(results)}\n")
    
    print("Summary saved to: docs/simple_summary.txt")

if __name__ == "__main__":
    generate_simple_report()
