"""
Utils package for ML Breast Cancer Analysis
Contains all modules and utilities for the project
"""

# Data processing functions
from .data_processor import load_and_explore_data, preprocess_data

# Model training functions
from .model_trainer import (
    train_and_evaluate_model, 
    analyze_feature_importance, 
    optimize_knn_k
)

# Visualization functions
from .visualizer import (
    plot_confusion_matrix, 
    plot_decision_boundary, 
    plot_feature_importance,
    plot_knn_analysis, 
    plot_svm_comparison, 
    plot_tree_models_comparison
)

# Model persistence functions
from .model_persistence import (
    save_model, 
    load_model, 
    save_all_models, 
    load_model_by_name
)

# Model comparison functions
from .model_comparison import (
    create_comparison_dataframe, 
    display_detailed_comparison,
    plot_comprehensive_comparison, 
    generate_model_summary_report,
    create_performance_radar_chart
)

# Medical analysis functions
from .medical_error_analysis import (
    analyze_medical_errors,
    plot_medical_error_analysis,
    generate_medical_recommendations,
    print_medical_recommendations
)

# CAP analysis functions
from .cap_analysis import (
    calculate_cap_analysis,
    assess_medical_model,
    plot_cap_analysis,
    print_cap_recommendations
)

__all__ = [
    # Data processing
    'load_and_explore_data',
    'preprocess_data',
    
    # Model training
    'train_and_evaluate_model',
    'analyze_feature_importance',
    'optimize_knn_k',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_decision_boundary',
    'plot_feature_importance',
    'plot_knn_analysis',
    'plot_svm_comparison',
    'plot_tree_models_comparison',
    
    # Model persistence
    'save_model',
    'load_model',
    'save_all_models',
    'load_model_by_name',
    
    # Model comparison
    'create_comparison_dataframe',
    'display_detailed_comparison',
    'plot_comprehensive_comparison',
    'generate_model_summary_report',
    'create_performance_radar_chart',
    
    # Medical analysis
    'analyze_medical_errors',
    'plot_medical_error_analysis',
    'generate_medical_recommendations',
    'print_medical_recommendations',
    
    # CAP analysis
    'calculate_cap_analysis',
    'assess_medical_model',
    'plot_cap_analysis',
    'print_cap_recommendations'
]
