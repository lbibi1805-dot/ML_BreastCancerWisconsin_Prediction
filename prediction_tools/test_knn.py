#!/usr/bin/env python3
"""
Simple KNN Cancer Prediction Test
=================================

Quick test script for the KNN breast cancer prediction model.
"""

import os
import numpy as np
import joblib
import json

def load_knn_model(models_dir="../Models"):
    """Load KNN model and metadata."""
    
    try:
        # Find KNN files
        knn_model_file = None
        knn_metadata_file = None
        
        for filename in os.listdir(models_dir):
            if filename.startswith('KNN') and filename.endswith('.joblib'):
                knn_model_file = filename
            elif filename.startswith('KNN') and filename.endswith('_metadata.json'):
                knn_metadata_file = filename
        
        if not knn_model_file or not knn_metadata_file:
            return None, None
        
        # Load model and metadata
        model_path = os.path.join(models_dir, knn_model_file)
        metadata_path = os.path.join(models_dir, knn_metadata_file)
        
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
        
    except Exception as e:
        print(f"Error loading KNN model: {e}")
        return None, None

def test_knn_prediction():
    """Test KNN model with sample data."""
    
    print("🔬 KNN BREAST CANCER PREDICTION TEST")
    print("="*50)
    
    # Load KNN model
    model, metadata = load_knn_model()
    
    if model is None:
        print("❌ KNN model not found")
        return
    
    print(f"✅ KNN Model loaded successfully")
    print(f"   📊 Test Accuracy: {metadata['results']['test_accuracy']:.4f}")
    print(f"   🎯 Algorithm: K-Nearest Neighbors (k=3)")
    
    # Test cases
    test_cases = [
        {
            'name': 'Benign Case',
            'data': [2, 1, 1, 1, 2, 1, 2, 1, 1],
            'expected': 'Benign'
        },
        {
            'name': 'Malignant Case', 
            'data': [8, 7, 8, 7, 6, 9, 7, 8, 3],
            'expected': 'Malignant'
        },
        {
            'name': 'Borderline Case',
            'data': [5, 3, 4, 3, 3, 5, 4, 4, 1],
            'expected': 'Unknown'
        }
    ]
    
    print(f"\n🧪 RUNNING TEST CASES:")
    print("-" * 50)
    
    feature_names = [
        'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
        'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
        'bland_chromatin', 'normal_nucleoli', 'mitoses'
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {test_case['name']}")
        print(f"   Expected: {test_case['expected']}")
        print(f"   Features: {test_case['data']}")
        
        # Make prediction
        X = np.array(test_case['data']).reshape(1, -1)
        
        try:
            prediction = model.predict(X)[0]
            diagnosis = "Benign" if prediction == 2 else "Malignant"
            
            # Get probabilities if available
            try:
                probs = model.predict_proba(X)[0]
                confidence = max(probs)
                print(f"   🎯 Prediction: {diagnosis} (value: {prediction})")
                print(f"   📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                print(f"   📈 Probabilities: Benign={probs[0]:.3f}, Malignant={probs[1]:.3f}")
            except:
                print(f"   🎯 Prediction: {diagnosis} (value: {prediction})")
                print(f"   📊 Probabilities not available")
            
            # Check if correct
            if test_case['expected'] != 'Unknown':
                correct = (diagnosis == test_case['expected'])
                status = "✅ CORRECT" if correct else "❌ INCORRECT"
                print(f"   {status}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ KNN Test completed!")
    print(f"🏥 Note: 2=Benign, 4=Malignant in model output")

if __name__ == "__main__":
    test_knn_prediction()
