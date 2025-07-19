"""
Visualization Module
Handles all visualization tasks including confusion matrices, decision boundaries, and comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, roc_auc_score


def plot_confusion_matrix(results, figsize=(8, 6)):
    """Plot confusion matrix for a model"""
    plt.figure(figsize=figsize)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix - {results["model_name"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_decision_boundary(model, model_name, X_train, y_train, feature_names, feature_indices=[0, 1], figsize=(10, 8)):
    """
    Plot decision boundary for 2D visualization
    Uses only first two features for visualization
    """
    # Use only selected features
    X_set = X_train[:, feature_indices]
    y_set = y_train
    
    # Create mesh
    h = 0.01
    x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Train model on 2D data
    model_2d = type(model)(**model.get_params()) if hasattr(model, 'get_params') else type(model)()
    model_2d.fit(X_set, y_set)
    
    # Plot decision boundary
    plt.figure(figsize=figsize)
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
    
    # Plot data points
    colors = ['#FA8072', '#1E90FF']
    for i, color in enumerate(colors):
        idx = np.where(y_set == i)
        plt.scatter(X_set[idx, 0], X_set[idx, 1], c=color, 
                   label=f'Class {i}', edgecolors='black', alpha=0.8)
    
    plt.xlabel(f'Feature {feature_indices[0]+1} ({feature_names[feature_indices[0]]})')
    plt.ylabel(f'Feature {feature_indices[1]+1} ({feature_names[feature_indices[1]]})')
    plt.title(f'{model_name} - Decision Boundary (2D Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_feature_importance(importance_df, model_name, figsize=(12, 6)):
    """Plot feature importance bar chart"""
    plt.figure(figsize=figsize)
    plt.bar(importance_df['Feature'], importance_df['Abs_Coefficient'] if 'Abs_Coefficient' in importance_df.columns else importance_df['Importance'])
    plt.title(f'{model_name} - Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_knn_analysis(k_results, knn_results, X_train, y_train, optimal_k):
    """Plot comprehensive KNN analysis"""
    k_df = pd.DataFrame(k_results, columns=['k', 'train_acc', 'test_acc', 'overfitting'])
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(k_df['k'], k_df['train_acc'], 'o-', label='Training Accuracy', color='blue')
    plt.plot(k_df['k'], k_df['test_acc'], 's-', label='Test Accuracy', color='red')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy vs K Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(k_df['k'], k_df['overfitting'], 'D-', color='green')
    plt.xlabel('K Value')
    plt.ylabel('Overfitting (Train - Test)')
    plt.title('KNN: Overfitting vs K Value')
    plt.grid(True, alpha=0.3)

    # Decision boundary comparison for different k values
    plt.subplot(2, 2, 3)
    X_subset = X_train[:, [0, 1]]
    y_subset = y_train

    from sklearn.neighbors import KNeighborsClassifier
    for i, k in enumerate([3, 5, 15]):
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_subset, y_subset)
        
        h = 0.1
        x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
        y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = knn_temp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0.5], colors=['red', 'blue', 'green'][i], 
                    linestyles=['-', '--', '-.'][i], linewidths=2, alpha=0.7)

    plt.scatter(X_subset[y_subset == 0, 0], X_subset[y_subset == 0, 1], c='red', alpha=0.6, label='Class 0')
    plt.scatter(X_subset[y_subset == 1, 0], X_subset[y_subset == 1, 1], c='blue', alpha=0.6, label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Decision Boundaries (k=3,5,15)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.bar(['Current (k=5)', f'Optimal (k={optimal_k})'], 
            [knn_results['test_accuracy'], k_df.loc[k_df['k']==optimal_k, 'test_acc'].values[0]],
            color=['lightblue', 'darkblue'])
    plt.ylabel('Test Accuracy')
    plt.title('Current vs Optimal K')
    plt.ylim(0.9, 1.0)

    plt.tight_layout()
    plt.show()


def plot_svm_comparison(svm_linear_results, svm_rbf_results):
    """Plot SVM comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Linear SVM confusion matrix
    plt.subplot(1, 3, 1)
    cm_linear = svm_linear_results['confusion_matrix']
    sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('SVM Linear - Confusion Matrix')

    # RBF SVM confusion matrix  
    plt.subplot(1, 3, 2)
    cm_rbf = svm_rbf_results['confusion_matrix']
    sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('SVM RBF - Confusion Matrix')

    # Metrics comparison
    plt.subplot(1, 3, 3)
    metrics_comparison = pd.DataFrame({
        'Linear': [svm_linear_results['test_accuracy'], svm_linear_results['precision'], 
                   svm_linear_results['recall'], svm_linear_results['f1_score']],
        'RBF': [svm_rbf_results['test_accuracy'], svm_rbf_results['precision'], 
                svm_rbf_results['recall'], svm_rbf_results['f1_score']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

    metrics_comparison.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'orange'])
    plt.title('SVM Kernels Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.ylim(0.9, 1.0)

    plt.tight_layout()
    plt.show()


def plot_tree_models_comparison(dt_importance, rf_importance, dt_results, rf_results):
    """Plot tree models comparison"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.bar(dt_importance['Feature'][:10], dt_importance['Importance'][:10])
    plt.title('Decision Tree - Top 10 Feature Importance')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(2, 2, 2)
    plt.bar(rf_importance['Feature'][:10], rf_importance['Importance'][:10])
    plt.title('Random Forest - Top 10 Feature Importance')
    plt.xticks(rotation=45, ha='right')

    # Tree-based models confusion matrices
    plt.subplot(2, 2, 3)
    cm_dt = dt_results['confusion_matrix']
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Decision Tree - Confusion Matrix')

    plt.subplot(2, 2, 4)
    cm_rf = rf_results['confusion_matrix']
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Random Forest - Confusion Matrix')

    plt.tight_layout()
    plt.show()
