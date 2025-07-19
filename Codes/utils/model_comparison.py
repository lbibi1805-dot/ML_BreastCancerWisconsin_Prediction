"""
Model Comparison Module
Handles comparison analysis between different ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def create_comparison_dataframe(all_results):
    """
    Create a comprehensive comparison dataframe from all model results
    
    Args:
        all_results: Dictionary containing results from all models
    
    Returns:
        DataFrame with model comparison metrics
    """
    comparison_data = []
    
    for model_name, result_data in all_results.items():
        if 'results' in result_data:
            results = result_data['results']
            
            # Extract training time if available
            training_time = results.get('training_time', 0)
            
            comparison_data.append({
                'Model': model_name,
                'Train_Accuracy': results.get('train_accuracy', 0),
                'Test_Accuracy': results.get('test_accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1_Score': results.get('f1_score', 0),
                'Training_Time': training_time,
                'Overfitting': results.get('train_accuracy', 0) - results.get('test_accuracy', 0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    
    return comparison_df


def display_detailed_comparison(comparison_df):
    """
    Display detailed comparison table with rankings
    """
    print("\n" + "="*100)
    print("🔍 DETAILED MODEL COMPARISON")
    print("="*100)
    
    # Display main comparison table
    print("\n📊 Performance Metrics:")
    print(comparison_df.to_string(index=False))
    
    # Best performing models for each metric
    print("\n🏆 BEST PERFORMERS BY METRIC:")
    print("-" * 50)
    
    metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'Training_Time']
    
    for metric in metrics:
        if metric == 'Training_Time':
            best_idx = comparison_df[metric].idxmin()  # Lowest time is best
            print(f"{metric:<15}: {comparison_df.loc[best_idx, 'Model']:<20} ({comparison_df.loc[best_idx, metric]:.4f}s)")
        else:
            best_idx = comparison_df[metric].idxmax()  # Highest score is best
            print(f"{metric:<15}: {comparison_df.loc[best_idx, 'Model']:<20} ({comparison_df.loc[best_idx, metric]:.4f})")
    
    # Overall ranking based on test accuracy
    print("\n🥇 OVERALL RANKING (by Test Accuracy):")
    print("-" * 40)
    ranked_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
        print(f"{i}. {row['Model']:<25} - {row['Test_Accuracy']:.4f}")


def plot_comprehensive_comparison(comparison_df, figsize=(16, 12)):
    """
    Create comprehensive comparison visualizations
    """
    plt.figure(figsize=figsize)
    
    # 1. Performance metrics comparison
    plt.subplot(2, 3, 1)
    metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, comparison_df[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x + width * 1.5, comparison_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0.85, 1.0)
    
    # 2. Test accuracy bar plot
    plt.subplot(2, 3, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))
    bars = plt.bar(comparison_df['Model'], comparison_df['Test_Accuracy'], color=colors)
    plt.title('Test Accuracy by Model')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.9, 1.0)
    
    # Add value labels on bars
    for bar, acc in zip(bars, comparison_df['Test_Accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Overfitting analysis
    plt.subplot(2, 3, 3)
    plt.bar(comparison_df['Model'], comparison_df['Overfitting'], 
            color=['red' if x > 0.05 else 'green' for x in comparison_df['Overfitting']])
    plt.title('Overfitting Analysis')
    plt.ylabel('Train - Test Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    plt.legend()
    
    # 4. Training time comparison
    plt.subplot(2, 3, 4)
    plt.bar(comparison_df['Model'], comparison_df['Training_Time'], color='skyblue')
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    
    # 5. Precision vs Recall scatter
    plt.subplot(2, 3, 5)
    plt.scatter(comparison_df['Precision'], comparison_df['Recall'], 
                s=100, c=comparison_df['F1_Score'], cmap='viridis', alpha=0.7)
    
    for i, model in enumerate(comparison_df['Model']):
        plt.annotate(model, (comparison_df.iloc[i]['Precision'], comparison_df.iloc[i]['Recall']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall (Color = F1-Score)')
    plt.colorbar(label='F1-Score')
    
    # 6. Model ranking heatmap
    plt.subplot(2, 3, 6)
    heatmap_data = comparison_df[['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']].T
    heatmap_data.columns = comparison_df['Model']
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, square=True)
    plt.title('Performance Heatmap')
    plt.ylabel('Metrics')
    
    plt.tight_layout()
    plt.show()


def generate_model_summary_report(comparison_df, all_results):
    """
    Generate a comprehensive text summary report
    """
    print("\n" + "="*80)
    print("📋 MODEL ANALYSIS SUMMARY REPORT")
    print("="*80)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Models Evaluated: {len(comparison_df)}")
    
    # Best overall model
    best_model = comparison_df.loc[comparison_df['Test_Accuracy'].idxmax()]
    print(f"\n🏆 BEST OVERALL MODEL: {best_model['Model']}")
    print(f"   Test Accuracy: {best_model['Test_Accuracy']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")
    print(f"   F1-Score: {best_model['F1_Score']:.4f}")
    
    # Performance tiers
    print("\n📈 PERFORMANCE TIERS:")
    high_performers = comparison_df[comparison_df['Test_Accuracy'] >= 0.97]
    good_performers = comparison_df[(comparison_df['Test_Accuracy'] >= 0.95) & (comparison_df['Test_Accuracy'] < 0.97)]
    avg_performers = comparison_df[comparison_df['Test_Accuracy'] < 0.95]
    
    print(f"   Excellent (≥97%): {', '.join(high_performers['Model'].tolist())}")
    print(f"   Good (95-97%): {', '.join(good_performers['Model'].tolist())}")
    print(f"   Average (<95%): {', '.join(avg_performers['Model'].tolist())}")
    
    # Overfitting analysis
    print("\n⚠️  OVERFITTING ANALYSIS:")
    overfitted = comparison_df[comparison_df['Overfitting'] > 0.05]
    if len(overfitted) > 0:
        print(f"   Models showing overfitting: {', '.join(overfitted['Model'].tolist())}")
    else:
        print("   No significant overfitting detected in any model")
    
    # Speed analysis
    print("\n⚡ TRAINING SPEED ANALYSIS:")
    fastest = comparison_df.loc[comparison_df['Training_Time'].idxmin()]
    slowest = comparison_df.loc[comparison_df['Training_Time'].idxmax()]
    print(f"   Fastest: {fastest['Model']} ({fastest['Training_Time']:.4f}s)")
    print(f"   Slowest: {slowest['Model']} ({slowest['Training_Time']:.4f}s)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print(f"   For Production: {best_model['Model']} (Best balance of accuracy and reliability)")
    
    if len(high_performers) > 0:
        print(f"   For Real-time Applications: {comparison_df.loc[comparison_df['Training_Time'].idxmin(), 'Model']} (Fastest training)")
    
    balanced_models = comparison_df[
        (comparison_df['Test_Accuracy'] >= 0.96) & 
        (comparison_df['Overfitting'] <= 0.03) &
        (comparison_df['Precision'] >= 0.95) &
        (comparison_df['Recall'] >= 0.95)
    ]
    
    if len(balanced_models) > 0:
        print(f"   Most Balanced: {balanced_models.iloc[0]['Model']} (Good accuracy with low overfitting)")
    
    print("\n" + "="*80)


def create_performance_radar_chart(comparison_df, figsize=(12, 8)):
    """
    Create radar chart comparing model performances
    """
    # Normalize metrics to 0-1 scale for radar chart
    metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    normalized_data = comparison_df[metrics].copy()
    
    # For training time, invert it (lower is better)
    if 'Training_Time' in comparison_df.columns:
        max_time = comparison_df['Training_Time'].max()
        normalized_data['Speed'] = 1 - (comparison_df['Training_Time'] / max_time)
        metrics.append('Speed')
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
    
    for i, (idx, row) in enumerate(comparison_df.iterrows()):
        values = []
        for metric in metrics:
            if metric == 'Speed':
                values.append(normalized_data.loc[idx, 'Speed'])
            else:
                values.append(row[metric])
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def export_comparison_results(comparison_df, all_results, filename=None):
    """
    Export comparison results to CSV and text files
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}"
    
    # Export CSV
    csv_file = f"{filename}.csv"
    comparison_df.to_csv(csv_file, index=False)
    
    # Export detailed report
    txt_file = f"{filename}_report.txt"
    with open(txt_file, 'w') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PERFORMANCE SUMMARY:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best performers
        f.write("BEST PERFORMERS:\n")
        f.write("-" * 20 + "\n")
        for metric in ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']:
            best_idx = comparison_df[metric].idxmax()
            f.write(f"{metric}: {comparison_df.loc[best_idx, 'Model']} ({comparison_df.loc[best_idx, metric]:.4f})\n")
    
    print(f"✅ Results exported:")
    print(f"   CSV: {csv_file}")
    print(f"   Report: {txt_file}")
    
    return csv_file, txt_file
