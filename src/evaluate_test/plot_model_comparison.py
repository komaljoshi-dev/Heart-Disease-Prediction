# src/evaluate_test/plot_model_comparison.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from features import y_test

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def plot_all_roc_curves(predictions_dict, y_true, save_path="reports/figures/all_models_roc.png"):
    """
    Plot ROC curves for all models in a single graph.
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, pred_data) in enumerate(predictions_dict.items()):
        y_prob = pred_data['y_prob']
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                    color=colors[i % len(colors)], 
                    lw=2.5, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', alpha=0.8, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved to: {save_path}")

def plot_all_pr_curves(predictions_dict, y_true, save_path="reports/figures/all_models_pr.png"):
    """
    Plot Precision-Recall curves for all models in a single graph.
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Calculate baseline (random classifier performance)
    baseline = np.sum(y_true) / len(y_true)
    
    for i, (model_name, pred_data) in enumerate(predictions_dict.items()):
        y_prob = pred_data['y_prob']
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, 
                    color=colors[i % len(colors)], 
                    lw=2.5,
                    label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    
    # Plot baseline
    plt.axhline(y=baseline, color='black', lw=1.5, linestyle='--', alpha=0.8, 
                label=f'Random Classifier (Baseline = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"PR curves saved to: {save_path}")

def plot_performance_comparison_bar(results_df, save_path="reports/figures/performance_comparison.png"):
    """
    Plot a bar chart comparing key metrics across all models.
    """
    # Select key metrics to plot
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity']
    
    # Prepare data
    plot_data = results_df[['Model'] + metrics_to_plot].set_index('Model')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot grouped bar chart
    x = np.arange(len(plot_data.index))
    width = 0.12
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot)/2) * width + width/2
        bars = ax.bar(x + offset, plot_data[metric], width, 
                     label=metric.upper(), color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Performance comparison saved to: {save_path}")

def create_summary_table(results_df):
    """
    Create a nicely formatted summary table.
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Sort by ROC-AUC (descending)
    sorted_df = results_df.sort_values('roc_auc', ascending=False).round(4)
    
    # Add ranking
    sorted_df.insert(0, 'Rank', range(1, len(sorted_df) + 1))
    
    # Display table
    print(sorted_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    print("Starting Model Comparison Plots...")
    
    # Load predictions and results
    try:
        predictions = joblib.load("reports/test_predictions.pkl")
        results_df = pd.read_csv("reports/test_set_evaluation.csv")
        
        print(f"Loaded predictions for {len(predictions)} models")
        print(f"Loaded evaluation results: {results_df.shape}")
        
        # Create all plots
        plot_all_roc_curves(predictions, y_test)
        plot_all_pr_curves(predictions, y_test)
        plot_performance_comparison_bar(results_df)
        
        # Display summary table
        create_summary_table(results_df)
        
        print("\nAll comparison plots created successfully!")
        print("Check the 'reports/figures/' directory for the plots")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run 'evaluate_test_set.py' first to generate the data!")
    except Exception as e:
        print(f"Unexpected error: {e}")
