#!/usr/bin/env python3
"""
Quick Test Script for Breast Cancer Prediction
==============================================

Simple script to quickly test trained models with sample data.
"""

import sys
import os
import numpy as np
import joblib
import json
from datetime import datetime

def load_model_from_files(model_name, models_dir="../Models"):
    """
    Load model directly from .joblib and .json files.
    
    Args:
        model_name (str): Name of the model to load
        models_dir (str): Directory containing model files
        
    Returns:
        tuple: (model, metadata) or (None, None) if not found
    """
    try:
        # Find the most recent model files for this model name
        model_files = []
        metadata_files = []
        
        for filename in os.listdir(models_dir):
            if filename.startswith(model_name) and filename.endswith('.joblib'):
                model_files.append(filename)
            elif filename.startswith(model_name) and filename.endswith('_metadata.json'):
                metadata_files.append(filename)
        
        if not model_files or not metadata_files:
            print(f"❌ No files found for model: {model_name}")
            return None, None
        
        # Use the most recent file (assuming timestamp in filename)
        model_file = sorted(model_files)[-1]
        metadata_file = sorted(metadata_files)[-1]
        
        # Load model
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(models_dir, metadata_file)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
        
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None, None

def quick_test():
    """Quick test with sample data."""
    
    print("🔄 Loading Random Forest model from Models folder...")
    
    # Load model directly from files
    model, metadata = load_model_from_files('Random Forest')
    
    if model is None:
        print("❌ Model not found. Available models in Models folder:")
        try:
            models_dir = "../Models"
            for filename in os.listdir(models_dir):
                if filename.endswith('.joblib'):
                    print(f"   - {filename}")
        except:
            print("   - Could not list Models directory")
        return
    
    print(f"✅ Model loaded! Test Accuracy: {metadata['results']['test_accuracy']:.4f}")
    
    # Sample patient data - IMPORTANT: These are RAW VALUES (not scaled yet)
    # We need to scale them like in training
    print("\n⚠️  Note: Using RAW feature values (will be processed by model)")
    
    # Sample patient data (benign case) - typical benign characteristics
    benign_sample = np.array([[2, 1, 1, 1, 2, 1, 2, 1, 1]])
    
    # Sample patient data (malignant case) - suspicious characteristics
    malignant_sample = np.array([[8, 7, 8, 7, 6, 9, 7, 8, 3]])
    
    print("\n🧪 TESTING PREDICTIONS:")
    print("="*50)
    
    # Feature names for reference
    feature_names = [
        'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
        'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 
        'bland_chromatin', 'normal_nucleoli', 'mitoses'
    ]
    
    print("📊 Feature order:", ", ".join(feature_names))
    print()
    
    # Test benign sample
    print("🔬 Sample 1 (Expected Benign):")
    print(f"   Features: {benign_sample[0]}")
    try:
        pred_benign = model.predict(benign_sample)[0]
        diagnosis_benign = "Benign" if pred_benign == 2 else "Malignant"
        print(f"   Prediction: {diagnosis_benign} (Raw value: {pred_benign})")
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            prob_benign = model.predict_proba(benign_sample)[0]
            print(f"   Probabilities: Benign={prob_benign[0]:.3f}, Malignant={prob_benign[1]:.3f}")
    except Exception as e:
        print(f"   ❌ Error in prediction: {e}")
    
    print()
    
    # Test malignant sample
    print("🔬 Sample 2 (Expected Malignant):")
    print(f"   Features: {malignant_sample[0]}")
    try:
        pred_malignant = model.predict(malignant_sample)[0]
        diagnosis_malignant = "Benign" if pred_malignant == 2 else "Malignant"
        print(f"   Prediction: {diagnosis_malignant} (Raw value: {pred_malignant})")
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            prob_malignant = model.predict_proba(malignant_sample)[0]
            print(f"   Probabilities: Benign={prob_malignant[0]:.3f}, Malignant={prob_malignant[1]:.3f}")
    except Exception as e:
        print(f"   ❌ Error in prediction: {e}")
    
    print("\n✅ Quick test completed!")
    print(f"🏥 Medical Note: In real applications, values 2=Benign, 4=Malignant")

def list_available_models():
    """List all available models in Models directory."""
    print("\n📁 Available Models:")
    try:
        models_dir = "../Models"
        model_names = set()
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                # Extract model name (everything before first underscore and timestamp)
                model_name = filename.split('_2025')[0]  # Remove timestamp
                model_names.add(model_name)
        
        for name in sorted(model_names):
            print(f"   - {name}")
    except Exception as e:
        print(f"   ❌ Error listing models: {e}")

if __name__ == "__main__":
    print("🏥 BREAST CANCER PREDICTION - QUICK TEST")
    print("="*50)
    
    # List available models first
    list_available_models()
    
    # Run quick test
    quick_test()
    
    print("\n" + "="*50)
    print("📝 Note: This test uses the Random Forest model")
    print("📁 Model files are loaded from: ../Models/")
    print("🔬 Test data represents typical benign vs malignant cell characteristics")
