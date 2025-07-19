"""
Model Training and Evaluation Module
Provides functions for training, evaluating, and analyzing ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_curve, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, verbose=True):
    """
    Train and evaluate a machine learning model
    
    Parameters:
    -----------
    model : sklearn estimator
        The machine learning model to train
    model_name : str
        Name of the model for display purposes
    X_train, X_test, y_train, y_test : arrays
        Training and testing data
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    results : dict
        Dictionary containing all evaluation metrics and predictions
    """
    print(f"\n{'='*60}")
    print(f"Training and Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Make predictions
    start_time = datetime.now()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    prediction_time = (datetime.now() - start_time).total_seconds()
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # ROC AUC (for binary classification)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_test_pred
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = None
    
    # Store results
    results = {
        'model': model,
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'overfitting': train_accuracy - test_accuracy
    }
    
    if verbose:
        print(f"Training Time: {training_time:.4f} seconds")
        print(f"Prediction Time: {prediction_time:.4f} seconds")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Overfitting: {results['overfitting']:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
    
    return results


def analyze_feature_importance(model, feature_names, model_name):
    """
    Analyze feature importance for different types of models
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
        
    Returns:
    --------
    importance_df : DataFrame
        DataFrame with feature importance information
    """
    print(f"\n{model_name} Feature Importance Analysis:")
    print("=" * 50)
    
    if hasattr(model, 'coef_'):
        # Linear models (Logistic Regression, SVM Linear)
        if len(model.coef_.shape) > 1:
            coefficients = model.coef_[0]
        else:
            coefficients = model.coef_
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("Top 5 Important Features (by absolute coefficient):")
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            print(f"{i+1}. {row['Feature']:25}: {row['Coefficient']:8.4f} (|{row['Abs_Coefficient']:.4f}|)")
            
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models (Decision Tree, Random Forest)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 Important Features:")
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            print(f"{i+1}. {row['Feature']:25}: {row['Importance']:.4f}")
            
    else:
        # Models without feature importance (KNN, Naive Bayes, SVM RBF)
        print(f"{model_name} does not provide feature importance information.")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': [1.0/len(feature_names)] * len(feature_names)
        })
    
    return importance_df


def optimize_knn_k(X_train, X_test, y_train, y_test, k_values=None, verbose=True):
    """
    Optimize K value for KNN classifier
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Training and testing data
    k_values : list, optional
        List of k values to test. Default: [3, 5, 7, 11, 15, 21, 25]
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    k_results : list
        List of tuples (k, train_acc, test_acc, overfitting)
    optimal_k : int
        Best k value based on test accuracy
    """
    if k_values is None:
        k_values = [3, 5, 7, 11, 15, 21, 25]
    
    if verbose:
        print("\nK-value Optimization:")
        print("=" * 25)
    
    k_results = []
    
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        train_acc = knn_temp.score(X_train, y_train)
        test_acc = knn_temp.score(X_test, y_test)
        overfitting = train_acc - test_acc
        k_results.append((k, train_acc, test_acc, overfitting))
        
        if verbose:
            print(f"k={k:2d}: Train={train_acc:.4f}, Test={test_acc:.4f}, Overfitting={overfitting:.4f}")
    
    # Find optimal k
    k_df = pd.DataFrame(k_results, columns=['k', 'train_acc', 'test_acc', 'overfitting'])
    optimal_k = k_df.loc[k_df['test_acc'].idxmax(), 'k']
    
    if verbose:
        print(f"\nOptimal k value: {optimal_k} (Test Accuracy: {k_df.loc[k_df['k']==optimal_k, 'test_acc'].values[0]:.4f})")
    
    return k_results, optimal_k
