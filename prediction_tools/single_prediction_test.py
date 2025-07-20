#!/usr/bin/env python3
"""
Single Prediction Test for Breast Cancer Classification Models
============================================================

This script demonstrates how to use the trained models to make predictions 
on new patient data for breast cancer diagnosis.

Features:
- Load saved models from ../Models/ directory  
- Make predictions on single patient samples
- Provide medical interpretation of results
- Calculate prediction confidence
- Test with sample patient data

Author: ML Breast Cancer Prediction Project
Date: July 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

class BreastCancerPredictor:
    """
    A class to handle breast cancer predictions using trained models.
    """
    
    def __init__(self, models_dir="../Models"):
        """
        Initialize the predictor with models directory.
        
        Args:
            models_dir (str): Path to directory containing saved models
        """
        self.models_dir = models_dir
        self.feature_names = [
            'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
            'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
            'bland_chromatin', 'normal_nucleoli', 'mitoses'
        ]
        self.models = {}
        
    def load_model_from_files(self, model_name):
        """
        Load a specific model from .joblib and .json files.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            tuple: (model, metadata) or (None, None) if not found
        """
        try:
            # Find the most recent model files for this model name
            model_files = []
            metadata_files = []
            
            for filename in os.listdir(self.models_dir):
                if filename.startswith(model_name) and filename.endswith('.joblib'):
                    model_files.append(filename)
                elif filename.startswith(model_name) and filename.endswith('_metadata.json'):
                    metadata_files.append(filename)
            
            if not model_files or not metadata_files:
                return None, None
            
            # Use the most recent file (assuming timestamp in filename)
            model_file = sorted(model_files)[-1]
            metadata_file = sorted(metadata_files)[-1]
            
            # Load model
            model_path = os.path.join(self.models_dir, model_file)
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, metadata_file)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            return None, None
    
    def load_model(self, model_name):
        """
        Load a specific trained model.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        model, metadata = self.load_model_from_files(model_name)
        
        if model and metadata:
            self.models[model_name] = {
                'model': model,
                'metadata': metadata
            }
            print(f"✅ {model_name} loaded successfully")
            print(f"   📊 Training Accuracy: {metadata['results']['test_accuracy']:.4f}")
            return True
        else:
            print(f"❌ Failed to load {model_name}")
            return False
    
    def load_all_models(self):
        """Load all available models."""
        # Get available model names from files
        model_names = set()
        try:
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.joblib'):
                    # Extract model name (everything before timestamp)
                    model_name = filename.split('_2025')[0]
                    model_names.add(model_name)
        except Exception as e:
            print(f"❌ Error reading models directory: {e}")
            return
        
        print("🔄 Loading all trained models...")
        for model_name in sorted(model_names):
            self.load_model(model_name)
        
        print(f"\n✅ Loaded {len(self.models)} models successfully")
    
    def preprocess_input(self, patient_data):
        """
        Convert patient data to the format expected by models.
        Note: Models were trained with scaled data, but they may have 
        their own preprocessing pipeline built-in.
        
        Args:
            patient_data (dict or list): Patient feature values
            
        Returns:
            numpy.ndarray: Data ready for prediction
        """
        if isinstance(patient_data, dict):
            # Convert dict to array in correct order
            data_array = np.array([patient_data[feature] for feature in self.feature_names])
        else:
            # Assume it's already a list/array
            data_array = np.array(patient_data)
        
        # Reshape for single sample
        data_array = data_array.reshape(1, -1)
        
        return data_array
    
    def predict_single(self, patient_data, model_name=None):
        """
        Make prediction for a single patient.
        
        Args:
            patient_data (dict or list): Patient feature values
            model_name (str): Specific model to use (if None, use all models)
            
        Returns:
            dict: Prediction results with medical interpretation
        """
        if not self.models:
            print("❌ No models loaded. Please load models first.")
            return None
        
        # Preprocess the input
        X_processed = self.preprocess_input(patient_data)
        
        results = {}
        
        # Use specific model or all models
        models_to_use = {model_name: self.models[model_name]} if model_name and model_name in self.models else self.models
        
        for name, model_data in models_to_use.items():
            model = model_data['model']
            
            try:
                # Make prediction
                prediction = model.predict(X_processed)[0]
                
                # Get prediction probability if available
                try:
                    probabilities = model.predict_proba(X_processed)[0]
                    confidence = max(probabilities)
                    # Assuming class 0 is benign (2) and class 1 is malignant (4)
                    prob_benign = probabilities[0]
                    prob_malignant = probabilities[1] 
                except:
                    confidence = None
                    prob_benign = None
                    prob_malignant = None
                
                # Medical interpretation
                diagnosis = "Benign (Non-cancerous)" if prediction == 2 else "Malignant (Cancerous)"
                risk_level = self._get_risk_level(confidence, prediction)
                
                results[name] = {
                    'prediction': int(prediction),
                    'diagnosis': diagnosis,
                    'confidence': confidence,
                    'prob_benign': prob_benign,
                    'prob_malignant': prob_malignant,
                    'risk_level': risk_level,
                    'test_accuracy': model_data['metadata']['results']['test_accuracy']
                }
                
            except Exception as e:
                print(f"❌ Error predicting with {name}: {e}")
                results[name] = {
                    'error': str(e)
                }
        
        return results
    
    def _get_risk_level(self, confidence, prediction):
        """Determine risk level based on confidence and prediction."""
        if confidence is None:
            return "Unknown"
        
        if prediction == 4:  # Malignant
            if confidence >= 0.9:
                return "High Risk - Immediate medical attention required"
            elif confidence >= 0.7:
                return "Moderate Risk - Further testing recommended"
            else:
                return "Low-Moderate Risk - Monitor closely"
        else:  # Benign
            if confidence >= 0.9:
                return "Low Risk - Routine monitoring"
            elif confidence >= 0.7:
                return "Low-Moderate Risk - Consider follow-up"
            else:
                return "Uncertain - Additional testing recommended"
    
    def predict_consensus(self, patient_data):
        """
        Get consensus prediction from all models.
        
        Args:
            patient_data (dict or list): Patient feature values
            
        Returns:
            dict: Consensus results with medical recommendation
        """
        if not self.models:
            print("❌ No models loaded. Please load models first.")
            return None
        
        results = self.predict_single(patient_data)
        
        if not results:
            return None
        
        # Filter out error results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No valid predictions obtained")
            return None
        
        # Count predictions
        malignant_count = sum(1 for r in valid_results.values() if r['prediction'] == 4)
        benign_count = len(valid_results) - malignant_count
        
        # Calculate average confidence for each class
        malignant_confidences = [r['prob_malignant'] for r in valid_results.values() if r['prob_malignant'] is not None]
        benign_confidences = [r['prob_benign'] for r in valid_results.values() if r['prob_benign'] is not None]
        
        avg_malignant_prob = np.mean(malignant_confidences) if malignant_confidences else 0
        avg_benign_prob = np.mean(benign_confidences) if benign_confidences else 0
        
        # Consensus decision
        consensus_prediction = 4 if malignant_count > benign_count else 2
        consensus_diagnosis = "Malignant (Cancerous)" if consensus_prediction == 4 else "Benign (Non-cancerous)"
        
        # Medical recommendation
        total_models = len(valid_results)
        if malignant_count >= total_models * 0.7:  # 70% or more predict malignant
            recommendation = "🚨 URGENT: High consensus for malignancy - Immediate medical evaluation required"
        elif malignant_count >= total_models * 0.3:  # 30-70% predict malignant
            recommendation = "⚠️ CAUTION: Mixed predictions - Additional testing strongly recommended"
        else:
            recommendation = "✅ REASSURING: Low risk indicated - Routine monitoring sufficient"
        
        consensus_result = {
            'consensus_prediction': consensus_prediction,
            'consensus_diagnosis': consensus_diagnosis,
            'malignant_votes': malignant_count,
            'benign_votes': benign_count,
            'total_models': total_models,
            'avg_malignant_probability': avg_malignant_prob,
            'avg_benign_probability': avg_benign_prob,
            'medical_recommendation': recommendation,
            'individual_results': valid_results
        }
        
        return consensus_result

def create_sample_patients():
    """Create sample patient data for testing."""
    
    patients = {
        'low_risk_patient': {
            'clump_thickness': 2,
            'uniform_cell_size': 1,
            'uniform_cell_shape': 1,
            'marginal_adhesion': 1,
            'single_epithelial_cell_size': 2,
            'bare_nuclei': 1,
            'bland_chromatin': 2,
            'normal_nucleoli': 1,
            'mitoses': 1,
            'description': 'Low risk patient - typical benign characteristics'
        },
        
        'high_risk_patient': {
            'clump_thickness': 8,
            'uniform_cell_size': 7,
            'uniform_cell_shape': 8,
            'marginal_adhesion': 7,
            'single_epithelial_cell_size': 6,
            'bare_nuclei': 9,
            'bland_chromatin': 7,
            'normal_nucleoli': 8,
            'mitoses': 3,
            'description': 'High risk patient - suspicious malignant characteristics'
        },
        
        'moderate_risk_patient': {
            'clump_thickness': 5,
            'uniform_cell_size': 4,
            'uniform_cell_shape': 5,
            'marginal_adhesion': 3,
            'single_epithelial_cell_size': 4,
            'bare_nuclei': 4,
            'bland_chromatin': 4,
            'normal_nucleoli': 3,
            'mitoses': 2,
            'description': 'Moderate risk patient - mixed characteristics'
        }
    }
    
    return patients

def print_prediction_results(patient_name, patient_data, results):
    """Print formatted prediction results."""
    
    print(f"\n" + "="*80)
    print(f"🩺 BREAST CANCER PREDICTION RESULTS: {patient_name.upper()}")
    print("="*80)
    
    # Patient information
    print(f"📋 Patient Description: {patient_data.get('description', 'N/A')}")
    print(f"📊 Feature Values:")
    for feature in ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                   'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                   'bland_chromatin', 'normal_nucleoli', 'mitoses']:
        print(f"   • {feature}: {patient_data[feature]}")
    
    if 'consensus_prediction' in results:
        # Consensus results
        print(f"\n🎯 CONSENSUS PREDICTION:")
        print(f"   • Diagnosis: {results['consensus_diagnosis']}")
        print(f"   • Votes: {results['malignant_votes']} Malignant, {results['benign_votes']} Benign")
        print(f"   • Average Malignant Probability: {results['avg_malignant_probability']:.3f}")
        print(f"   • Average Benign Probability: {results['avg_benign_probability']:.3f}")
        print(f"\n💡 MEDICAL RECOMMENDATION:")
        print(f"   {results['medical_recommendation']}")
        
        print(f"\n📋 INDIVIDUAL MODEL RESULTS:")
        print("-" * 80)
        print(f"{'Model':<20} | {'Diagnosis':<25} | {'Confidence':<10} | {'Risk Level'}")
        print("-" * 80)
        for model_name, result in results['individual_results'].items():
            if 'error' not in result:
                confidence_str = f"{result['confidence']:.3f}" if result['confidence'] else "N/A"
                print(f"{model_name:<20} | {result['diagnosis']:<25} | {confidence_str:<10} | {result['risk_level']}")
    else:
        # Single model results
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"\n🔬 {model_name} RESULTS:")
                print(f"   • Diagnosis: {result['diagnosis']}")
                print(f"   • Confidence: {result['confidence']:.3f}" if result['confidence'] else "   • Confidence: N/A")
                print(f"   • Risk Level: {result['risk_level']}")
                print(f"   • Model Test Accuracy: {result['test_accuracy']:.3f}")

def main():
    """Main function to demonstrate single prediction testing."""
    
    print("🏥 BREAST CANCER PREDICTION SYSTEM")
    print("="*50)
    print("This system uses trained ML models to predict breast cancer diagnosis")
    print("based on cell characteristics from fine needle aspirate samples.\n")
    
    # Initialize predictor
    predictor = BreastCancerPredictor()
    
    # Load all models
    predictor.load_all_models()
    
    if not predictor.models:
        print("❌ No models available. Please train models first.")
        print("💡 Run the Jupyter notebook ml_models_comparison.ipynb to train models")
        return
    
    # Get sample patients
    patients = create_sample_patients()
    
    # Test with each sample patient
    for patient_name, patient_data in patients.items():
        # Make consensus prediction
        consensus_results = predictor.predict_consensus(patient_data)
        
        if consensus_results:
            print_prediction_results(patient_name, patient_data, consensus_results)
    
    # Interactive prediction option
    print(f"\n" + "="*80)
    print("🔬 INTERACTIVE PREDICTION MODE")
    print("="*80)
    print("You can now enter custom patient data for prediction.")
    print("Commands: 'demo' | 'custom' | 'q' (quit)")
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 'demo':
                # Show sample prediction
                patient_data = patients['moderate_risk_patient']
                results = predictor.predict_consensus(patient_data)
                print_prediction_results('demo_patient', patient_data, results)
            elif user_input == 'custom':
                print("\nEnter patient feature values (1-10 scale):")
                
                custom_patient = {}
                for feature in predictor.feature_names:
                    while True:
                        try:
                            value = input(f"{feature} (1-10): ")
                            value = float(value)
                            if 1 <= value <= 10:
                                custom_patient[feature] = value
                                break
                            else:
                                print("Please enter a value between 1 and 10")
                        except ValueError:
                            print("Please enter a valid number")
                
                # Make prediction
                results = predictor.predict_consensus(custom_patient)
                custom_patient['description'] = 'Custom patient data'
                print_prediction_results('custom_patient', custom_patient, results)
            else:
                print("Invalid command. Use 'demo', 'custom', or 'q'")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Prediction system terminated. Thank you!")

if __name__ == "__main__":
    main()
