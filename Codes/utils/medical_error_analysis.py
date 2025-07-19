"""
Medical Error Analysis Module
Analyzes Type I and Type II errors for medical diagnosis applications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_medical_errors(model_results):
    """
    Analyze Type I and Type II errors for medical diagnosis
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model evaluation results with confusion matrices
        
    Returns:
    --------
    dict : Dictionary containing error analysis for each model
    """
    error_analysis = {}
    
    for model_name, results in model_results.items():
        cm = results['confusion_matrix']
        
        # In medical context: Class 2=Benign (Negative), Class 4=Malignant (Positive)
        # CM format: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        
        total_benign = tn + fp
        total_malignant = fn + tp
        total_predictions = tn + fp + fn + tp
        
        # Calculate error rates
        type1_error_rate = fp / total_benign if total_benign > 0 else 0  # False Positive Rate
        type2_error_rate = fn / total_malignant if total_malignant > 0 else 0  # False Negative Rate
        
        # Medical-specific metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for cancer detection
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        
        error_analysis[model_name] = {
            'Type_I_Error_Rate': type1_error_rate,
            'Type_II_Error_Rate': type2_error_rate,
            'Type_I_Count': fp,
            'Type_II_Count': fn,
            'Sensitivity': sensitivity,  # Critical for cancer detection
            'Specificity': specificity,
            'Total_Patients': total_predictions
        }
    
    return error_analysis


def plot_medical_error_analysis(error_df):
    """
    Create comprehensive medical error analysis visualizations
    
    Parameters:
    -----------
    error_df : pandas.DataFrame
        DataFrame containing error analysis results
        
    Returns:
    --------
    None : Displays plots
    """
    # Visualize error comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Error Rates Comparison
    axes[0,0].bar(error_df.index, error_df['Type_I_Error_Rate'], alpha=0.7, color='orange', label='Type I (False Positive)')
    axes[0,0].bar(error_df.index, error_df['Type_II_Error_Rate'], alpha=0.7, color='red', label='Type II (False Negative)')
    axes[0,0].set_title('Medical Error Rates Comparison', fontsize=14, weight='bold')
    axes[0,0].set_ylabel('Error Rate')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    # 2. Error Counts
    x_pos = np.arange(len(error_df))
    width = 0.35
    axes[0,1].bar(x_pos - width/2, error_df['Type_I_Count'], width, label='Type I Count', color='orange', alpha=0.7)
    axes[0,1].bar(x_pos + width/2, error_df['Type_II_Count'], width, label='Type II Count', color='red', alpha=0.7)
    axes[0,1].set_title('Medical Error Counts', fontsize=14, weight='bold')
    axes[0,1].set_ylabel('Number of Errors')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(error_df.index, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Sensitivity vs Specificity
    axes[1,0].scatter(error_df['Specificity'], error_df['Sensitivity'], s=100, alpha=0.7)
    for i, model in enumerate(error_df.index):
        axes[1,0].annotate(model, (error_df['Specificity'].iloc[i], error_df['Sensitivity'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1,0].set_xlabel('Specificity (True Negative Rate)')
    axes[1,0].set_ylabel('Sensitivity (True Positive Rate)')
    axes[1,0].set_title('Sensitivity vs Specificity', fontsize=14, weight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # 4. Medical Risk Assessment
    risk_scores = error_df['Type_II_Error_Rate'] * 3 + error_df['Type_I_Error_Rate']  # Weight Type II more heavily
    axes[1,1].barh(error_df.index, risk_scores, color=['red' if score > 0.1 else 'orange' if score > 0.05 else 'green' for score in risk_scores])
    axes[1,1].set_title('Medical Risk Score (Type II × 3 + Type I)', fontsize=14, weight='bold')
    axes[1,1].set_xlabel('Risk Score (Lower is Better)')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_medical_recommendations(error_df):
    """
    Generate medical recommendations based on error analysis
    
    Parameters:
    -----------
    error_df : pandas.DataFrame
        DataFrame containing error analysis results
        
    Returns:
    --------
    dict : Dictionary containing medical recommendations
    """
    # Medical recommendations
    best_sensitivity = error_df['Sensitivity'].idxmax()
    best_specificity = error_df['Specificity'].idxmax()
    lowest_type2 = error_df['Type_II_Error_Rate'].idxmin()
    
    recommendations = {
        'best_sensitivity': {
            'model': best_sensitivity,
            'sensitivity': error_df.loc[best_sensitivity, 'Sensitivity'],
            'type2_error_rate': error_df.loc[best_sensitivity, 'Type_II_Error_Rate']
        },
        'best_specificity': {
            'model': best_specificity,
            'specificity': error_df.loc[best_specificity, 'Specificity'],
            'type1_error_rate': error_df.loc[best_specificity, 'Type_I_Error_Rate']
        },
        'safest_model': {
            'model': lowest_type2,
            'type2_error_rate': error_df.loc[lowest_type2, 'Type_II_Error_Rate'],
            'missed_cancers': error_df.loc[lowest_type2, 'Type_II_Count']
        }
    }
    
    return recommendations


def print_medical_recommendations(recommendations):
    """
    Print formatted medical recommendations
    
    Parameters:
    -----------
    recommendations : dict
        Dictionary containing medical recommendations
    """
    print("\n🩺 MEDICAL RECOMMENDATIONS:")
    print("=" * 50)
    
    best_sens = recommendations['best_sensitivity']
    print(f"🎯 Best for Cancer Detection (Highest Sensitivity): {best_sens['model']}")
    print(f"   Sensitivity: {best_sens['sensitivity']:.4f}")
    print(f"   Type II Error Rate: {best_sens['type2_error_rate']:.4f}")

    best_spec = recommendations['best_specificity']
    print(f"\n🛡️  Best for Avoiding False Alarms (Highest Specificity): {best_spec['model']}")
    print(f"   Specificity: {best_spec['specificity']:.4f}")
    print(f"   Type I Error Rate: {best_spec['type1_error_rate']:.4f}")

    safest = recommendations['safest_model']
    print(f"\n⚠️  Safest Model (Lowest Type II Error): {safest['model']}")
    print(f"   Type II Error Rate: {safest['type2_error_rate']:.4f}")
    print(f"   Missed Cancer Cases: {safest['missed_cancers']}")

    print(f"\n🏥 MEDICAL CONCLUSION:")
    print(f"For cancer screening, prioritize models with:")
    print(f"1. Lowest Type II Error Rate (minimize missed cancers)")
    print(f"2. High Sensitivity (detect cancer cases)")
    print(f"3. Acceptable Type I Error Rate (manageable false alarms)")
