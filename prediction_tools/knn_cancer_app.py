#!/usr/bin/env python3
"""
KNN Breast Cancer Prediction Application
========================================

Simple application using only the trained KNN model for breast cancer prediction.
This is a focused implementation using the chosen KNN (k=3) model.

Features:
- Load only KNN model from ../Models/ directory
- Make predictions using KNN algorithm
- Medical interpretation of results
- Simple and clean interface

Author: ML Breast Cancer Prediction Project
Date: July 2025
"""

import os
import numpy as np
import joblib
import json

class KNNBreastCancerPredictor:
    """
    Breast cancer predictor using only KNN model.
    """
    
    def __init__(self, models_dir="../Models"):
        """
        Initialize the KNN predictor.
        
        Args:
            models_dir (str): Path to directory containing saved models
        """
        self.models_dir = models_dir
        self.feature_names = [
            'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
            'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
            'bland_chromatin', 'normal_nucleoli', 'mitoses'
        ]
        self.knn_model = None
        self.model_metadata = None
        
    def load_knn_model(self):
        """
        Load the trained KNN model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find KNN model files
            knn_model_file = None
            knn_metadata_file = None
            
            for filename in os.listdir(self.models_dir):
                if filename.startswith('KNN') and filename.endswith('.joblib'):
                    knn_model_file = filename
                elif filename.startswith('KNN') and filename.endswith('_metadata.json'):
                    knn_metadata_file = filename
            
            if not knn_model_file or not knn_metadata_file:
                print("❌ KNN model files not found in Models directory")
                return False
            
            # Load KNN model
            model_path = os.path.join(self.models_dir, knn_model_file)
            self.knn_model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, knn_metadata_file)
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            print("✅ KNN Model loaded successfully!")
            print(f"   📊 Model: K-Nearest Neighbors (k=3)")
            print(f"   🎯 Test Accuracy: {self.model_metadata['results']['test_accuracy']:.4f}")
            print(f"   📈 F1-Score: {self.model_metadata['results']['f1_score']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading KNN model: {e}")
            return False
    
    def predict(self, patient_data):
        """
        Make prediction for a single patient using KNN.
        
        Args:
            patient_data (dict or list): Patient feature values (1-10 scale)
            
        Returns:
            dict: Prediction results with medical interpretation
        """
        if self.knn_model is None:
            print("❌ KNN model not loaded. Please load model first.")
            return None
        
        # Convert input to numpy array
        if isinstance(patient_data, dict):
            # Convert dict to array in correct order
            data_array = np.array([patient_data[feature] for feature in self.feature_names])
        else:
            # Assume it's already a list/array
            data_array = np.array(patient_data)
        
        # Reshape for single prediction
        X = data_array.reshape(1, -1)
        
        try:
            # Make prediction
            prediction = self.knn_model.predict(X)[0]
            
            # Get prediction probabilities (if available)
            try:
                probabilities = self.knn_model.predict_proba(X)[0]
                prob_benign = probabilities[0]    # Class 2 (Benign)
                prob_malignant = probabilities[1] # Class 4 (Malignant)
                confidence = max(probabilities)
            except:
                prob_benign = None
                prob_malignant = None
                confidence = None
            
            # Medical interpretation
            diagnosis = "Benign (Non-cancerous)" if prediction == 2 else "Malignant (Cancerous)"
            medical_advice = self._get_medical_advice(prediction, confidence)
            
            result = {
                'prediction_value': int(prediction),
                'diagnosis': diagnosis,
                'confidence': confidence,
                'prob_benign': prob_benign,
                'prob_malignant': prob_malignant,
                'medical_advice': medical_advice,
                'model_accuracy': self.model_metadata['results']['test_accuracy'],
                'algorithm': 'K-Nearest Neighbors (k=3)'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def _get_medical_advice(self, prediction, confidence):
        """
        Provide medical advice based on prediction and confidence.
        
        Args:
            prediction (int): Model prediction (2 or 4)
            confidence (float): Prediction confidence
            
        Returns:
            str: Medical advice
        """
        if confidence is None:
            return "⚠️ Consult healthcare professional for proper evaluation"
        
        if prediction == 4:  # Malignant
            if confidence >= 0.9:
                return "🚨 HIGH RISK: Seek immediate medical attention - likely malignant"
            elif confidence >= 0.7:
                return "⚠️ MODERATE RISK: Schedule urgent medical consultation"
            else:
                return "⚠️ UNCERTAIN: Additional testing recommended"
        else:  # Benign
            if confidence >= 0.9:
                return "✅ LOW RISK: Routine monitoring recommended"
            elif confidence >= 0.7:
                return "✅ REASSURING: Consider regular check-ups"
            else:
                return "⚠️ UNCERTAIN: Follow-up testing advised"

def create_sample_cases():
    """Create sample test cases for different risk levels."""
    
    cases = {
        'typical_benign': {
            'clump_thickness': 2,
            'uniform_cell_size': 1,
            'uniform_cell_shape': 1,
            'marginal_adhesion': 1,
            'single_epithelial_cell_size': 2,
            'bare_nuclei': 1,
            'bland_chromatin': 2,
            'normal_nucleoli': 1,
            'mitoses': 1,
            'description': 'Typical benign case - low values across all features'
        },
        
        'suspicious_malignant': {
            'clump_thickness': 8,
            'uniform_cell_size': 7,
            'uniform_cell_shape': 8,
            'marginal_adhesion': 7,
            'single_epithelial_cell_size': 6,
            'bare_nuclei': 9,
            'bland_chromatin': 7,
            'normal_nucleoli': 8,
            'mitoses': 3,
            'description': 'Suspicious malignant case - high values in key features'
        },
        
        'borderline_case': {
            'clump_thickness': 5,
            'uniform_cell_size': 3,
            'uniform_cell_shape': 4,
            'marginal_adhesion': 3,
            'single_epithelial_cell_size': 3,
            'bare_nuclei': 5,
            'bland_chromatin': 4,
            'normal_nucleoli': 4,
            'mitoses': 1,
            'description': 'Borderline case - mixed feature values'
        }
    }
    
    return cases

def print_patient_info(case_name, patient_data):
    """Print patient information in a formatted way."""
    
    print(f"\n" + "="*70)
    print(f"🩺 PATIENT CASE: {case_name.upper().replace('_', ' ')}")
    print("="*70)
    print(f"📋 Description: {patient_data['description']}")
    print(f"📊 Cell Features (Scale 1-10):")
    
    for feature in ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                   'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                   'bland_chromatin', 'normal_nucleoli', 'mitoses']:
        value = patient_data[feature]
        risk_indicator = "🔴" if value >= 7 else "🟡" if value >= 4 else "🟢"
        print(f"   {risk_indicator} {feature}: {value}")

def print_prediction_result(result):
    """Print prediction result in a formatted way."""
    
    if not result:
        print("❌ No prediction result available")
        return
    
    print(f"\n🔬 KNN PREDICTION RESULT:")
    print("-" * 40)
    print(f"🎯 Diagnosis: {result['diagnosis']}")
    print(f"📊 Raw Prediction: {result['prediction_value']} ({'Benign=2, Malignant=4'})")
    
    if result['confidence']:
        print(f"🎲 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        
    if result['prob_benign'] and result['prob_malignant']:
        print(f"📈 Probabilities:")
        print(f"   • Benign: {result['prob_benign']:.3f} ({result['prob_benign']*100:.1f}%)")
        print(f"   • Malignant: {result['prob_malignant']:.3f} ({result['prob_malignant']*100:.1f}%)")
    
    print(f"\n💡 MEDICAL ADVICE:")
    print(f"   {result['medical_advice']}")
    
    print(f"\n📋 MODEL INFO:")
    print(f"   • Algorithm: {result['algorithm']}")
    print(f"   • Test Accuracy: {result['model_accuracy']:.3f} ({result['model_accuracy']*100:.1f}%)")

def interactive_prediction(predictor):
    """Interactive mode for custom patient data."""
    
    print(f"\n" + "="*70)
    print("🔬 INTERACTIVE PREDICTION MODE")
    print("="*70)
    print("Enter patient cell feature values (scale 1-10)")
    print("Commands: 'demo' (show sample) | 'predict' (enter data) | 'q' (quit)")
    
    sample_cases = create_sample_cases()
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                break
                
            elif command == 'demo':
                # Show a demo case
                case_name = 'borderline_case'
                patient_data = sample_cases[case_name]
                
                print_patient_info(case_name, patient_data)
                result = predictor.predict(patient_data)
                print_prediction_result(result)
                
            elif command == 'predict':
                print("\nEnter feature values (1-10):")
                
                custom_data = {}
                for feature in predictor.feature_names:
                    while True:
                        try:
                            value = input(f"  {feature} (1-10): ").strip()
                            value = float(value)
                            if 1 <= value <= 10:
                                custom_data[feature] = value
                                break
                            else:
                                print("    ⚠️ Please enter a value between 1 and 10")
                        except ValueError:
                            print("    ⚠️ Please enter a valid number")
                
                # Make prediction
                custom_data['description'] = 'Custom patient data'
                print_patient_info('custom_patient', custom_data)
                result = predictor.predict(custom_data)
                print_prediction_result(result)
                
            else:
                print("Invalid command. Use 'demo', 'predict', or 'q'")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main application function."""
    
    print("🏥 KNN BREAST CANCER PREDICTION APPLICATION")
    print("="*60)
    print("Using K-Nearest Neighbors (k=3) algorithm for breast cancer diagnosis")
    print("Based on Wisconsin Breast Cancer Dataset")
    print()
    
    # Initialize predictor
    predictor = KNNBreastCancerPredictor()
    
    # Load KNN model
    if not predictor.load_knn_model():
        print("❌ Failed to load KNN model. Please ensure:")
        print("   1. Models directory exists: ../Models/")
        print("   2. KNN model files are present")
        print("   3. Run the training notebook first")
        return
    
    # Test with sample cases
    print(f"\n🧪 TESTING WITH SAMPLE CASES:")
    sample_cases = create_sample_cases()
    
    for case_name, patient_data in sample_cases.items():
        print_patient_info(case_name, patient_data)
        result = predictor.predict(patient_data)
        print_prediction_result(result)
    
    # Interactive mode
    interactive_prediction(predictor)
    
    print(f"\n" + "="*60)
    print("✅ KNN Prediction Application terminated")
    print("⚠️  DISCLAIMER: For research purposes only - not for medical diagnosis")
    print("🏥 Always consult healthcare professionals for medical decisions")

if __name__ == "__main__":
    main()
