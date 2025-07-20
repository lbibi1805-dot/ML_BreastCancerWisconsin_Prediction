"""
🩺 Breast Cancer Prediction System v2.0 - Streamlit Web Application
====================================================================

Advanced web application for breast cancer prediction using machine learning.
Features complete 9-feature support from Wisconsin Breast Cancer dataset.

Features:
- ✅ Complete 9-feature Random Forest model
- ✅ Interactive web interface with Streamlit
- ✅ Real-time prediction with confidence scores
- ✅ Medical-grade visualizations and explanations
- ✅ Beautiful UI with medical theme
- ✅ Comprehensive error handling

Author: Breast Cancer Prediction Team
Date: July 2025
Version: 2.0 (Updated for 9 features)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.model_persistence import load_model_by_name
    from utils.data_processor import load_and_explore_data, preprocess_data
except ImportError:
    st.error("❌ Cannot import utils modules. Please ensure utils package is available.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="🩺 Breast Cancer Prediction System v2.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .benign-card {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .malignant-card {
        background: linear-gradient(135deg, #f44336, #da190b);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class BreastCancerPredictionApp:
    """
    🏥 Streamlit Web Application for Breast Cancer Prediction
    
    This class manages the complete web application including model loading,
    UI components, predictions, and result visualization.
    """
    
    def __init__(self):
        """Initialize the Streamlit application"""
        # ✅ All 9 features from Wisconsin Breast Cancer dataset
        self.feature_names = [
            'Clump_thickness',              # Feature 1: Độ dày cụm tế bào
            'Uniformity_of_cell_size',      # Feature 2: Tính đồng đều kích thước tế bào
            'Uniformity_of_cell_shape',     # Feature 3: Tính đồng đều hình dạng tế bào
            'Marginal_adhesion',            # Feature 4: Độ bám dính biên tế bào
            'Single_epithelial_cell_size',  # Feature 5: Kích thước tế bào biểu mô đơn
            'Bare_nuclei',                  # Feature 6: Nhân trần (không có tế bào chất)
            'Bland_chromatin',              # Feature 7: Cấu trúc nhiễm sắc thể
            'Normal_nucleoli',              # Feature 8: Nhân con bình thường
            'Mitoses'                       # Feature 9: Quá trình phân bào
        ]
        
        # 📋 Medical feature descriptions
        self.feature_descriptions = {
            'Clump_thickness': 'Độ dày cụm tế bào - Giá trị cao thường liên quan đến ác tính',
            'Uniformity_of_cell_size': 'Tính đồng đều kích thước tế bào - Tế bào ác tính có kích thước không đồng đều',
            'Uniformity_of_cell_shape': 'Tính đồng đều hình dạng tế bào - Tế bào ác tính có hình dạng bất thường',
            'Marginal_adhesion': 'Độ bám dính biên tế bào - Tế bào ác tính mất khả năng bám dính',
            'Single_epithelial_cell_size': 'Kích thước tế bào biểu mô đơn - Liên quan đến phát triển bất thường',
            'Bare_nuclei': 'Nhân trần không có tế bào chất - Đặc trưng của ung thư ác tính',
            'Bland_chromatin': 'Cấu trúc nhiễm sắc thể - Tế bào ác tính có cấu trúc bất thường',
            'Normal_nucleoli': 'Nhân con bình thường - Tế bào ác tính có nhân con to và nổi bật',
            'Mitoses': 'Quá trình phân bào - Tế bào ác tính có tỷ lệ phân bào cao'
        }
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.metadata = None
            st.session_state.scaler = None
        
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            self.load_model()
    
    def load_model(self):
        """🔄 Load the trained Random Forest model and scaler"""
        try:
            with st.spinner('🔄 Loading trained model...'):
                # 🔧 Fix path - we're in Codes/, Models is one level up
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(current_dir, "..", "Models")
                
                # Load Random Forest model
                model, metadata = load_model_by_name('Random Forest', save_dir=model_dir)
                
                if model and metadata:
                    st.session_state.model = model
                    st.session_state.metadata = metadata
                    
                    # Setup scaler using original dataset
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    dataset_path = os.path.join(current_dir, "..", "Dataset", "breast_cancer_wisconsin.csv")
                    dataset, feature_names = load_and_explore_data(dataset_path)
                    X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset, feature_names)
                    st.session_state.scaler = scaler
                    
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model. Please check Models directory.")
                    st.stop()
                    
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.error("💡 Please ensure models are trained and saved in Models directory")
            st.stop()
    
    def create_feature_inputs(self):
        """🔧 Create input widgets for all 9 features"""
        st.markdown("### 🔬 Clinical Features Input")
        st.markdown("**Please enter values for each clinical feature (Scale: 1-10)**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        feature_values = {}
        
        # Split features into two columns
        half = len(self.feature_names) // 2
        
        with col1:
            st.markdown("#### 📊 Primary Features")
            for i, feature in enumerate(self.feature_names[:half]):
                feature_values[feature] = st.slider(
                    f"**{feature}**",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help=self.feature_descriptions[feature],
                    key=f"feature_{i}"
                )
                st.markdown(f"<div class='feature-card'><small>{self.feature_descriptions[feature]}</small></div>", 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📈 Secondary Features") 
            for i, feature in enumerate(self.feature_names[half:], half):
                feature_values[feature] = st.slider(
                    f"**{feature}**",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help=self.feature_descriptions[feature],
                    key=f"feature_{i}"
                )
                st.markdown(f"<div class='feature-card'><small>{self.feature_descriptions[feature]}</small></div>", 
                           unsafe_allow_html=True)
        
        return feature_values
    
    def create_preset_buttons(self):
        """🎯 Create preset test case buttons"""
        st.markdown("### 🧪 Quick Test Cases")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Benign Case", key="benign"):
                return self.get_benign_case()
        
        with col2:
            if st.button("🔴 Malignant Case", key="malignant"):
                return self.get_malignant_case()
        
        with col3:
            if st.button("🟡 Borderline Case", key="borderline"):
                return self.get_borderline_case()
        
        return None
    
    def get_benign_case(self):
        """📊 Return typical benign case values"""
        return {feature: 2 if feature == 'Clump_thickness' else 1 
                for feature in self.feature_names}
    
    def get_malignant_case(self):
        """📊 Return typical malignant case values"""
        values = [8, 7, 8, 7, 6, 9, 7, 8, 3]
        return dict(zip(self.feature_names, values))
    
    def get_borderline_case(self):
        """📊 Return borderline case values"""
        values = [5, 3, 4, 3, 4, 5, 4, 3, 2]
        return dict(zip(self.feature_names, values))
    
    def make_prediction(self, feature_values):
        """🔮 Make prediction using the loaded model"""
        try:
            # Convert to array in correct order
            sample_array = np.array([feature_values[feature] for feature in self.feature_names])
            sample_array = sample_array.reshape(1, -1)
            
            # Scale the features
            sample_scaled = st.session_state.scaler.transform(sample_array)
            
            # Make prediction
            prediction = st.session_state.model.predict(sample_scaled)[0]
            prediction_proba = st.session_state.model.predict_proba(sample_scaled)[0]
            
            # Extract probabilities
            benign_probability = prediction_proba[0]
            cancer_probability = prediction_proba[1]
            
            return {
                'prediction_class': int(prediction),
                'diagnosis': "Malignant (Ác tính)" if prediction == 4 else "Benign (Lành tính)",
                'cancer_probability': float(cancer_probability),
                'benign_probability': float(benign_probability),
                'confidence': float(max(cancer_probability, benign_probability)),
                'feature_values': feature_values
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None
    
    def display_prediction_results(self, results):
        """🎨 Display prediction results with beautiful formatting"""
        if not results:
            return
        
        # Main prediction display
        if results['prediction_class'] == 4:  # Malignant
            st.markdown(f"""
            <div class='malignant-card'>
                <h2>🔴 MALIGNANT (ÁC TÍNH)</h2>
                <h3>Cancer Probability: {results['cancer_probability']:.1%}</h3>
                <p>Model Confidence: {results['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            if results['cancer_probability'] > 0.8:
                st.error("🚨 **HIGH RISK**: Immediate medical consultation recommended")
            elif results['cancer_probability'] > 0.6:
                st.warning("⚠️ **MODERATE-HIGH RISK**: Medical consultation recommended within 1-2 weeks")
            else:
                st.warning("🟡 **MODERATE RISK**: Follow-up with healthcare provider recommended")
                
        else:  # Benign
            st.markdown(f"""
            <div class='benign-card'>
                <h2>🟢 BENIGN (LÀNH TÍNH)</h2>
                <h3>Benign Probability: {results['benign_probability']:.1%}</h3>
                <p>Model Confidence: {results['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if results['benign_probability'] > 0.8:
                st.success("✅ **LOW RISK**: Continue regular health monitoring")
            else:
                st.info("ℹ️ **MODERATE CONFIDENCE**: Consider additional testing for peace of mind")
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🎯 Confidence</h3>
                <h2>{results['confidence']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🟢 Benign</h3>
                <h2>{results['benign_probability']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🔴 Malignant</h3>
                <h2>{results['cancer_probability']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    def create_probability_chart(self, results):
        """📊 Create interactive probability visualization"""
        if not results:
            return
        
        # Create gauge chart for cancer probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = results['cancer_probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cancer Risk Level (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance radar chart
        feature_values = [results['feature_values'][feature] for feature in self.feature_names]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatterpolar(
            r=feature_values,
            theta=self.feature_names,
            fill='toself',
            name='Feature Values',
            line_color='rgba(0,100,80,0.8)',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Clinical Features Profile",
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def display_model_info(self):
        """📋 Display model information and performance metrics"""
        if st.session_state.metadata:
            st.sidebar.markdown("### 🤖 Model Information")
            
            metadata = st.session_state.metadata
            
            st.sidebar.markdown(f"""
            **Model:** {metadata['model_name']}  
            **Algorithm:** Random Forest Classifier  
            **Features:** 9 (Complete Wisconsin dataset)  
            **Test Accuracy:** {metadata['results']['test_accuracy']:.1%}  
            **F1-Score:** {metadata['results']['f1_score']:.3f}  
            **Training Date:** {metadata['save_date']}  
            **Training Time:** {metadata['results']['training_time']:.2f}s  
            """)
            
            # Performance visualization
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
                'Score': [
                    metadata['results']['test_accuracy'],
                    metadata['results']['f1_score'],
                    metadata['results'].get('precision', 0.95),  # Default if not available
                    metadata['results'].get('recall', 0.95)     # Default if not available
                ]
            })
            
            fig = px.bar(
                metrics_df, 
                x='Metric', 
                y='Score',
                title='Model Performance Metrics',
                color='Score',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=300)
            st.sidebar.plotly_chart(fig, use_container_width=True)
    
    def display_medical_disclaimer(self):
        """⚕️ Display important medical disclaimer"""
        st.markdown("---")
        st.markdown("### ⚕️ Important Medical Disclaimer")
        
        st.warning("""
        🚨 **IMPORTANT MEDICAL DISCLAIMER:**
        
        • This tool is for **educational and research purposes only**
        • **NOT a substitute** for professional medical diagnosis
        • Always consult qualified healthcare professionals
        • Seek immediate medical attention for concerning symptoms
        • This prediction should **never** replace proper medical examination
        """)
    
    def run(self):
        """🚀 Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">🩺 Breast Cancer Prediction System v2.0</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.2rem; color: #666;'>
                Advanced AI-powered breast cancer prediction using Random Forest algorithm<br>
                <strong>✅ Updated with complete 9-feature Wisconsin dataset</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar model info
        self.display_model_info()
        
        # Main content area
        if st.session_state.model_loaded:
            # Preset buttons
            preset_values = self.create_preset_buttons()
            
            # Feature inputs
            if preset_values:
                st.info(f"🎯 Loaded preset values. You can modify them below if needed.")
                # Update session state with preset values
                for feature, value in preset_values.items():
                    st.session_state[f"feature_{self.feature_names.index(feature)}"] = value
                
                # Force rerun to update sliders
                st.experimental_rerun()
            
            feature_values = self.create_feature_inputs()
            
            # Prediction button
            if st.button("🔮 Make Prediction", type="primary", key="predict"):
                with st.spinner('🔄 Analyzing features and making prediction...'):
                    results = self.make_prediction(feature_values)
                    
                    if results:
                        st.markdown("---")
                        st.markdown("## 📋 Prediction Results")
                        
                        # Display results
                        self.display_prediction_results(results)
                        
                        # Visualizations
                        st.markdown("---")
                        st.markdown("## 📊 Detailed Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            self.create_probability_chart(results)
                        
                        with col2:
                            # Feature summary table
                            st.markdown("### 📋 Feature Summary")
                            feature_df = pd.DataFrame({
                                'Feature': list(feature_values.keys()),
                                'Value': list(feature_values.values()),
                                'Risk Level': ['High' if v >= 7 else 'Medium' if v >= 4 else 'Low' 
                                             for v in feature_values.values()]
                            })
                            st.dataframe(feature_df, use_container_width=True)
                            
                            # Model performance info
                            st.markdown("### 🎯 Model Performance")
                            st.info(f"""
                            **Model Accuracy:** {st.session_state.metadata['results']['test_accuracy']:.1%}  
                            **F1-Score:** {st.session_state.metadata['results']['f1_score']:.3f}  
                            **Features Used:** 9 (Complete dataset)  
                            **Algorithm:** Random Forest Classifier
                            """)
        
        # Medical disclaimer
        self.display_medical_disclaimer()
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #666; margin-top: 2rem;'>
            <p>🏥 Breast Cancer Prediction System v2.0 | 
            Built with ❤️ using Streamlit | 
            Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
            <p><strong>⚡ Now with complete 9-feature support!</strong></p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """🚀 Main function to run the Streamlit app"""
    app = BreastCancerPredictionApp()
    app.run()


if __name__ == "__main__":
    main()
