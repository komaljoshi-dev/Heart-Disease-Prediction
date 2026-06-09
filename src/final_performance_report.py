"""
Final Performance Report Generator
Comprehensive evaluation confirming >85% accuracy requirement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_evaluation_results():
    """Load all evaluation results from previous analyses"""
    
    results_data = {}
    
    # Load main evaluation results
    if os.path.exists("reports/test_set_evaluation.csv"):
        results_data['test_evaluation'] = pd.read_csv("reports/test_set_evaluation.csv")
    
    # Load pairwise comparison results
    if os.path.exists("reports/model_pairwise_stats.csv"):
        results_data['pairwise_stats'] = pd.read_csv("reports/model_pairwise_stats.csv")
    
    # Load CV results if available
    if os.path.exists("reports/cv_results.csv"):
        results_data['cv_results'] = pd.read_csv("reports/cv_results.csv")
    
    return results_data

def generate_executive_summary(results_data):
    """Generate executive summary of model performance"""
    
    test_results = results_data['test_evaluation']
    
    # Find best performing model
    best_model = test_results.loc[test_results['accuracy'].idxmax()]
    
    # Count models exceeding 85% accuracy threshold
    models_above_85 = test_results[test_results['accuracy'] >= 0.85]
    
    summary = {
        'project_name': 'Heart Disease Prediction System',
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_models_evaluated': len(test_results),
        'models_above_85_percent': len(models_above_85),
        'requirement_met': len(models_above_85) > 0,
        'best_model': {
            'name': best_model['Model'],
            'accuracy': float(best_model['accuracy']),
            'precision': float(best_model['precision']),
            'recall': float(best_model['recall']),
            'f1_score': float(best_model['f1']),
            'roc_auc': float(best_model['roc_auc']),
            'specificity': float(best_model['specificity'])
        },
        'models_above_threshold': models_above_85[['Model', 'accuracy', 'roc_auc']].to_dict('records')
    }
    
    return summary

def create_performance_dashboard(results_data, save_path="reports/final_performance_dashboard.png"):
    """Create comprehensive performance dashboard"""
    
    test_results = results_data['test_evaluation']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy Comparison Bar Chart
    ax1 = fig.add_subplot(gs[0, :2])
    models = test_results['Model']
    accuracies = test_results['accuracy']
    
    colors = ['red' if acc < 0.85 else 'green' for acc in accuracies]
    bars = ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.7)
    
    # Add 85% threshold line
    ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.8, label='85% Threshold')
    
    ax1.set_title('Model Accuracy vs 85% Requirement', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. ROC-AUC Performance
    ax2 = fig.add_subplot(gs[0, 2:])
    roc_aucs = test_results['roc_auc']
    
    bars2 = ax2.bar(range(len(models)), roc_aucs, color='skyblue', alpha=0.7)
    ax2.set_title('ROC-AUC Performance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    for bar, auc in zip(bars2, roc_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2., auc + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Precision-Recall Comparison
    ax3 = fig.add_subplot(gs[1, :2])
    
    x = np.arange(len(models))
    width = 0.35
    
    bars3 = ax3.bar(x - width/2, test_results['precision'], width, label='Precision', alpha=0.7)
    bars4 = ax3.bar(x + width/2, test_results['recall'], width, label='Recall', alpha=0.7)
    
    ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 and Specificity
    ax4 = fig.add_subplot(gs[1, 2:])
    
    bars5 = ax4.bar(x - width/2, test_results['f1'], width, label='F1-Score', alpha=0.7)
    bars6 = ax4.bar(x + width/2, test_results['specificity'], width, label='Specificity', alpha=0.7)
    
    ax4.set_title('F1-Score vs Specificity', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Summary Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create summary table data
    table_data = []
    for _, row in test_results.iterrows():
        meets_req = "✓" if row['accuracy'] >= 0.85 else "✗"
        table_data.append([
            row['Model'],
            f"{row['accuracy']:.1%}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['roc_auc']:.3f}",
            meets_req
        ])
    
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', '≥85%']
    
    table = ax5.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on accuracy requirement
    for i in range(1, len(table_data) + 1):
        if table_data[i-1][6] == "✓":  # Meets requirement
            table[(i, 6)].set_facecolor('#90EE90')  # Light green
        else:
            table[(i, 6)].set_facecolor('#FFB6C1')  # Light red
    
    ax5.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Heart Disease Prediction Model - Final Performance Report', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save the dashboard
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance dashboard saved to: {save_path}")

def generate_detailed_report(results_data, summary, save_path="reports/final_performance_report.json"):
    """Generate detailed JSON report"""
    
    test_results = results_data['test_evaluation']
    
    # Detailed analysis
    detailed_report = {
        'meta_information': {
            'report_title': 'Heart Disease Prediction System - Final Performance Report',
            'generated_date': datetime.now().isoformat(),
            'project_phase': 'Final Evaluation',
            'accuracy_requirement': '≥85%',
            'requirement_status': 'MET' if summary['requirement_met'] else 'NOT MET'
        },
        
        'executive_summary': summary,
        
        'detailed_model_performance': [],
        
        'statistical_analysis': {},
        
        'clinical_relevance': {
            'sensitivity_analysis': {},
            'specificity_analysis': {},
            'clinical_recommendations': []
        },
        
        'deployment_readiness': {
            'models_ready_for_deployment': [],
            'recommended_model': summary['best_model']['name'],
            'risk_categorization': 'Implemented',
            'probability_calibration': 'Available',
            'ui_integration': 'Ready'
        }
    }
    
    # Add detailed performance for each model
    for _, model_row in test_results.iterrows():
        model_performance = {
            'model_name': model_row['Model'],
            'performance_metrics': {
                'accuracy': float(model_row['accuracy']),
                'precision': float(model_row['precision']),
                'recall': float(model_row['recall']),
                'f1_score': float(model_row['f1']),
                'roc_auc': float(model_row['roc_auc']),
                'specificity': float(model_row['specificity']),
                'pr_auc': float(model_row['pr_auc']) if pd.notna(model_row['pr_auc']) else None
            },
            'meets_accuracy_requirement': bool(model_row['accuracy'] >= 0.85),
            'clinical_suitability': 'High' if model_row['accuracy'] >= 0.90 else 'Medium' if model_row['accuracy'] >= 0.85 else 'Low',
            'recommended_for_deployment': bool(model_row['accuracy'] >= 0.85)
        }
        detailed_report['detailed_model_performance'].append(model_performance)
    
    # Statistical analysis
    detailed_report['statistical_analysis'] = {
        'mean_accuracy': float(test_results['accuracy'].mean()),
        'std_accuracy': float(test_results['accuracy'].std()),
        'max_accuracy': float(test_results['accuracy'].max()),
        'min_accuracy': float(test_results['accuracy'].min()),
        'models_above_90_percent': int(len(test_results[test_results['accuracy'] >= 0.90])),
        'models_above_85_percent': int(len(test_results[test_results['accuracy'] >= 0.85]))
    }
    
    # Clinical relevance analysis
    best_sensitivity = test_results.loc[test_results['recall'].idxmax()]
    best_specificity = test_results.loc[test_results['specificity'].idxmax()]
    
    detailed_report['clinical_relevance']['sensitivity_analysis'] = {
        'best_sensitivity_model': best_sensitivity['Model'],
        'best_sensitivity_value': float(best_sensitivity['recall']),
        'clinical_importance': 'High sensitivity reduces false negatives (missed diagnoses)'
    }
    
    detailed_report['clinical_relevance']['specificity_analysis'] = {
        'best_specificity_model': best_specificity['Model'],
        'best_specificity_value': float(best_specificity['specificity']),
        'clinical_importance': 'High specificity reduces false positives (unnecessary anxiety/procedures)'
    }
    
    # Clinical recommendations
    detailed_report['clinical_relevance']['clinical_recommendations'] = [
        f"Primary recommendation: Use {summary['best_model']['name']} for highest overall accuracy ({summary['best_model']['accuracy']:.1%})",
        "Implement risk categorization system for clinical decision support",
        "Use calibrated probabilities for reliable risk assessment",
        "Consider ensemble approach for critical cases",
        "Regular model monitoring and retraining recommended"
    ]
    
    # Deployment readiness
    ready_models = [model['model_name'] for model in detailed_report['detailed_model_performance'] 
                   if model['meets_accuracy_requirement']]
    detailed_report['deployment_readiness']['models_ready_for_deployment'] = ready_models
    
    # Save detailed report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(detailed_report, f, indent=2, default=str)
    
    print(f"Detailed report saved to: {save_path}")
    return detailed_report

def print_executive_summary(summary):
    """Print executive summary to console"""
    
    print("="*80)
    print("HEART DISEASE PREDICTION SYSTEM - FINAL PERFORMANCE REPORT")
    print("="*80)
    
    print(f"\nProject: {summary['project_name']}")
    print(f"Evaluation Date: {summary['evaluation_date']}")
    print(f"Total Models Evaluated: {summary['total_models_evaluated']}")
    
    print(f"\n🎯 ACCURACY REQUIREMENT (≥85%): {'✅ MET' if summary['requirement_met'] else '❌ NOT MET'}")
    print(f"Models Meeting Requirement: {summary['models_above_85_percent']}/{summary['total_models_evaluated']}")
    
    print(f"\n🏆 BEST PERFORMING MODEL:")
    best = summary['best_model']
    print(f"  Model: {best['name']}")
    print(f"  Accuracy: {best['accuracy']:.1%}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  F1-Score: {best['f1_score']:.3f}")
    print(f"  ROC-AUC: {best['roc_auc']:.3f}")
    print(f"  Specificity: {best['specificity']:.3f}")
    
    print(f"\n📊 MODELS EXCEEDING 85% ACCURACY:")
    if summary['models_above_threshold']:
        for i, model in enumerate(summary['models_above_threshold'], 1):
            print(f"  {i}. {model['Model']}: {model['accuracy']:.1%} (ROC-AUC: {model['roc_auc']:.3f})")
    else:
        print("  None")
    
    print("\n" + "="*80)

def generate_final_performance_report():
    """Main function to generate complete final performance report"""
    
    print("Generating Final Performance Report...")
    
    # Load all evaluation data
    results_data = load_evaluation_results()
    
    if 'test_evaluation' not in results_data:
        print("Error: No test evaluation results found. Please run the evaluation pipeline first.")
        return None
    
    # Generate executive summary
    summary = generate_executive_summary(results_data)
    
    # Print executive summary
    print_executive_summary(summary)
    
    # Create performance dashboard
    create_performance_dashboard(results_data)
    
    # Generate detailed report
    detailed_report = generate_detailed_report(results_data, summary)
    
    # Create summary text file for quick reference
    with open("reports/executive_summary.txt", "w") as f:
        f.write("HEART DISEASE PREDICTION SYSTEM - EXECUTIVE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Evaluation Date: {summary['evaluation_date']}\n")
        f.write(f"Total Models: {summary['total_models_evaluated']}\n")
        f.write(f"Accuracy Requirement: ≥85%\n")
        f.write(f"Requirement Status: {'MET' if summary['requirement_met'] else 'NOT MET'}\n")
        f.write(f"Models Meeting Requirement: {summary['models_above_85_percent']}\n\n")
        f.write(f"Best Model: {summary['best_model']['name']}\n")
        f.write(f"Best Accuracy: {summary['best_model']['accuracy']:.1%}\n")
        f.write(f"Best ROC-AUC: {summary['best_model']['roc_auc']:.3f}\n")
    
    print(f"\nFinal Performance Report Generation Complete!")
    print(f"Files generated:")
    print(f"  - Executive Summary: reports/executive_summary.txt")
    print(f"  - Detailed Report: reports/final_performance_report.json")
    print(f"  - Performance Dashboard: reports/final_performance_dashboard.png")
    
    return detailed_report, summary

if __name__ == "__main__":
    detailed_report, summary = generate_final_performance_report()
