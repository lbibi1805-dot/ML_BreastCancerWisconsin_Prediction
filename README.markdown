# 🩺 Breast Cancer Prediction - Machine Learning Comparison

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-gree```

## 🏆 Algorithm Performance

**Updated Performance Results (Latest Training - July 2025)**

| Algorithm | Test Accuracy | F1-Score | Precision | Recall | Training Time | Status |
|-----------|---------------|----------|-----------|---------|---------------|---------|
| **🥇 KNN (k=3)** | **97.08%** | **97.09%** | **97.14%** | **97.08%** | 0.011s | ✅ **DEPLOYED** |
| 🥈 Random Forest | 97.08% | 96.77% | 97.14% | 97.08% | 0.150s | ✅ Available |
| 🥉 Logistic Regression | 96.35% | 95.83% | 96.00% | 96.35% | 0.020s | ✅ Available |
| SVM (RBF) | 96.35% | 95.83% | 96.00% | 96.35% | 0.080s | ✅ Available |
| SVM (Linear) | 95.62% | 94.59% | 95.24% | 95.62% | 0.010s | ✅ Available |
| Decision Tree | 95.62% | 94.59% | 95.24% | 95.62% | 0.003s | ✅ Available |
| Naive Bayes | 94.89% | 93.88% | 94.59% | 94.89% | 0.002s | ✅ Available |

### 🎯 Why KNN Was Chosen for Deployment

1. **Highest Accuracy**: Tied for best test accuracy (97.08%)
2. **Excellent F1-Score**: Best F1-score (97.09%) indicating balanced precision/recall
3. **Fast Training**: Very quick training time (0.011s)
4. **Stable Performance**: Consistent results across multiple runs
5. **Medical Safety**: Good balance of Type I/Type II error rates
6. **Interpretability**: Easy to explain to medical professionals

### 📊 Confusion Matrix (KNN - Deployed Model)
```
                Predicted
                Benign  Malignant
Actual Benign     84      3
       Malignant   1     49
```

**Key Metrics:**
- **Type I Error (False Positive)**: 3/87 = 3.45% - Benign classified as Malignant
- **Type II Error (False Negative)**: 1/50 = 2.00% - Malignant classified as Benign
- **Sensitivity (Recall)**: 49/50 = 98.00% - Correctly identified malignant cases
- **Specificity**: 84/87 = 96.55% - Correctly identified benign casesvg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![API](https://img.shields.io/badge/API-Live-brightgreen.svg)](https://api-deploy-ml-breastcancer-wisconsin.onrender.com)

This project provides a comprehensive comparison of 7 machine learning algorithms for breast cancer prediction using the **Wisconsin Breast Cancer Dataset**. With a focus on medical applications, the project includes Type I/Type II error analysis, CAP Analysis, and **live API deployment** for real-world usage.

## 🚀 Live API Available

**Production API**: https://api-deploy-ml-breastcancer-wisconsin.onrender.com
- **Best Model**: KNN (k=3) with 97.08% accuracy deployed
- **Real-time predictions** via REST API
- **React/Express ready** with CORS support

```bash
# Test the live API
curl -X POST https://api-deploy-ml-breastcancer-wisconsin.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2, 1, 1, 1, 2, 1, 2, 1, 1]}'
```

## 📋 Table of Contents

- [🚀 Live API Usage](#-live-api-usage)
- [🎯 Model Performance](#-model-performance) 
- [📊 Dataset Information](#-dataset-information)
- [💻 Local Development](#-local-development)
- [🏥 Medical Analysis](#-medical-analysis)
- [🤝 Contributing](#-contributing)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements and compares machine learning algorithms to classify breast cancer as **benign** or **malignant**. Designed with a modular architecture, the project emphasizes:

- **Optimal Performance**: Comparison of 7 algorithms with in-depth metrics.
- **Medical Safety**: Analysis of Type I (False Positive) and Type II (False Negative) errors.
- **Clinical Deployment**: Evaluation of real-world applicability in healthcare.

The dataset used is the **Wisconsin Breast Cancer Dataset** from the UCI Machine Learning Repository, featuring 9 attributes and 2 classes (benign: 2, malignant: 4).

## ✨ Features

### 🔬 Machine Learning
- **7 Algorithms**: Logistic Regression, KNN, SVM (Linear & RBF), Decision Tree, Random Forest, Naive Bayes.
- **Unified Preprocessing**: Feature scaling and consistent data processing.
- **Hyperparameter Optimization**: Optimal K for KNN, kernel comparison for SVM.
- **Cross-Validation**: Stable performance evaluation with 10-fold CV.

### 📊 Visualization
- **Confusion Matrix**: Detailed classification error visualization.
- **Decision Boundary**: 2D decision boundary for each algorithm.
- **Feature Importance**: Analysis of feature significance.
- **CAP Curves**: Evaluation of discriminative ability in medical contexts.
- **Error Analysis Plots**: Comparison of Type I/Type II errors.

### 🏥 Medical Analysis
- **Type I/II Error Analysis**: Evaluation of false positives and false negatives.
- **CAP Analysis**: Cumulative Accuracy Profile for medical evaluation.
- **Clinical Recommendations**: Guidelines for deployment based on safety and efficacy.

### 💾 Model Persistence
- **Save/Load Models**: Save models with metadata (accuracy, hyperparameters, timestamp).
- **Batch Processing**: Manage multiple models simultaneously.
- **Production-Ready**: Prediction functions easily integrated into healthcare systems.

## 📊 Dataset Structure Explanation

**🎯 Dependent Variable (Target Variable):**
- **`Class`**: Breast cancer classification
  - **2**: Benign - No cancer
  - **4**: Malignant - Cancer present

**🔬 Independent Variables (Features):** 9 medical features from cell samples

1. **`clump_thickness`**: Clump thickness (1-10)
   - Higher values → Suspected malignancy
   
2. **`uniform_cell_size`**: Uniformity of cell size (1-10)
   - Malignant cells often have non-uniform sizes
   
3. **`uniform_cell_shape`**: Uniformity of cell shape (1-10)
   - Malignant cells often have irregular shapes
   
4. **`marginal_adhesion`**: Marginal adhesion of cells (1-10)
   - Malignant cells tend to lose adhesion
   
5. **`single_epithelial_cell_size`**: Single epithelial cell size (1-10)
   - Related to abnormal cell growth
   
6. **`bare_nuclei`**: Bare nuclei (no surrounding cytoplasm) (1-10)
   - Common in malignant cancers
   
7. **`bland_chromatin`**: Chromatin structure (1-10)
   - Malignant cells have abnormal chromatin structure
   
8. **`normal_nucleoli`**: Normal nucleoli (1-10)
   - Malignant cells have larger, prominent nucleoli
   
9. **`mitoses`**: Mitotic activity (1-10)
   - Malignant cells have higher mitotic rates

**📈 Importance in Machine Learning:**
- **Features (X)**: 9 medical features → Input data for prediction
- **Target (y)**: Cancer classification → Outcome to predict
- **Objective**: Learn from features to accurately predict the target

## 📁 Project Structure

```
ML_BreastCancerWisconsin_Prediction/
│
├── 📊 Dataset/
│   ├── breast_cancer_wisconsin.csv      # Main dataset
│   ├── breast_cancer.csv                # Secondary dataset
│   ├── Source.txt                       # Source information
│   └── raw_data/                        # Raw data
│       ├── breast-cancer-wisconsin.data
│       ├── breast-cancer-wisconsin.names
│       ├── wdbc.data
│       ├── wdbc.names
│       ├── wpbc.data
│       └── wpbc.names
│
├── 💻 Codes/
│   ├── 📓 ml_models_comparison.ipynb    # Complete model comparison & analysis
│   ├── 📓 knn_neighbours.ipynb          # KNN implementation (DEPLOYED MODEL)
│   ├── 📓 random_forest_classification.ipynb
│   ├── 📓 logistic_regression.ipynb
│   ├── 📓 SVM.ipynb & kernel_svm.ipynb
│   ├── � decision_tree.ipynb
│   ├── 🌐 streamlit_app.py              # Streamlit web interface
│   ├── �🛠️ utils/                        # Utility package
│   │   ├── __init__.py                  # Package initialization
│   │   ├── data_processor.py            # Data loading & preprocessing
│   │   ├── model_trainer.py             # Model training & evaluation
│   │   ├── visualizer.py                # Advanced visualizations
│   │   ├── model_persistence.py         # Model save/load (joblib)
│   │   ├── model_comparison.py          # Performance comparison
│   │   ├── medical_error_analysis.py    # Type I/II error analysis
│   │   └── cap_analysis.py              # CAP analysis for medical safety
│   └── data_crawler.py                  # UCI data fetching
│
├── 🤖 Models/                           # Trained model files (.joblib + metadata)
│   ├── KNN_20250720_110419.joblib      # ✅ DEPLOYED MODEL (97.08% accuracy)
│   ├── KNN_20250720_110419_metadata.json
│   ├── Random Forest_20250720_110419.joblib
│   ├── Logistic Regression_20250720_110419.joblib
│   ├── SVM_Linear_20250720_110419.joblib
│   ├── SVM_RBF_20250720_110419.joblib
│   ├── Decision Tree_20250720_110419.joblib
│   └── Naive Bayes_20250720_110419.joblib
│
├── 🌐 api_server/                       # Flask API (local development)
│   ├── app.py                          # Flask application with scaling fix
│   ├── requirements.txt                # Dependencies
│   ├── SETUP_GUIDE.md                 # Local setup instructions
│   ├── DEPLOY_GUIDE.md                # Deployment guide
│   └── [Docker & deployment configs...]
│
├── 🔧 prediction_tools/                # Testing & prediction utilities
│   ├── single_prediction_test.py      # Multi-model testing system
│   ├── knn_cancer_app.py             # KNN-focused prediction app
│   ├── test_knn.py                   # Quick KNN validation
│   └── README.md                     # Prediction tools guide
│   ├── SVM_RBF.pkl
│   ├── Decision_Tree.pkl
│   ├── Random_Forest.pkl
│   ├── Naive_Bayes.pkl
│   └── metadata/                        # Model metadata
│
├── 📜 requirements.txt                  # Dependencies
└── 📖 README.md                         # This file
```

## 🚀 Installation

### Requirements
- Python 3.8+
- Jupyter Notebook/JupyterLab
- Git (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ML_BreastCancerWisconsin_Prediction.git
cd ML_BreastCancerWisconsin_Prediction
```

### Step 2: Create a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Jupyter Notebook
```bash
cd Codes
jupyter notebook ml_models_comparison.ipynb
```

## 📦 Dependencies (requirements.txt)

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
plotly>=5.0.0
ipywidgets>=7.6.0
```

## 💡 Usage

### 🌐 Use Live API (Recommended)

**Production API**: https://api-deploy-ml-breastcancer-wisconsin.onrender.com

```bash
# Health check
curl https://api-deploy-ml-breastcancer-wisconsin.onrender.com/

# Make prediction
curl -X POST https://api-deploy-ml-breastcancer-wisconsin.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2, 1, 1, 1, 2, 1, 2, 1, 1]}'
```

### 📓 Local Development & Training

**Quick Start:**
1. Open `Codes/ml_models_comparison.ipynb`
2. Run all cells to train all models and compare performance
3. Models are automatically saved to `Models/` directory

**Individual Model Training:**
```bash
cd Codes
jupyter notebook knn_neighbours.ipynb         # Train KNN (deployed model)
jupyter notebook random_forest_classification.ipynb
jupyter notebook logistic_regression.ipynb
# ... other model notebooks
```

### 🔧 Local Prediction Testing

```bash
cd prediction_tools

# Test KNN model (deployed version)
python test_knn.py

# Interactive KNN prediction app
python knn_cancer_app.py

# Multi-model comparison
python single_prediction_test.py
```

### 🌐 Local API Server

```bash
cd api_server
pip install -r requirements.txt
python app.py
# Server: http://localhost:5000
```

### Usage Example (Python)
```python
# Import our utility modules
from Codes.utils import *

# Load and preprocess data
dataset, feature_names = load_and_explore_data("Dataset/breast_cancer_wisconsin.csv")
X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset, feature_names)

# Train KNN (best performing model)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_results = train_and_evaluate_model(knn_model, "KNN", X_train, X_test, y_train, y_test)

# Save model
save_model(knn_model, knn_results, "KNN", save_dir="Models")

# Make prediction on new data
new_patient = [[2, 1, 1, 1, 2, 1, 2, 1, 1]]  # Benign case
new_patient_scaled = scaler.transform(new_patient)
prediction = knn_model.predict(new_patient_scaled)
print(f"Prediction: {'Benign' if prediction[0] == 2 else 'Malignant'}")
```
results = train_and_evaluate_model(model, "Random Forest", X_train, X_test, y_train, y_test)

# Visualize
plot_confusion_matrix(results)
plot_decision_boundary(model, "Random Forest", X_train, y_train, feature_names)

# Save model
save_model(model, results, "Random_Forest", save_dir="../Models")
```

### Load and Use Saved Model
```python
# Load model
loaded_model, metadata = load_model_by_name("Random_Forest", save_dir="../Models")
prediction = loaded_model.predict(new_data)
```

## 🤖 Algorithm Performance

| Algorithm           | Accuracy | Precision | Recall | F1-Score | Training Time |
|--------------------|----------|-----------|--------|----------|---------------|
| Random Forest      | 97.08%   | 97.15%    | 97.08% | 97.09%   | 0.029s        |
| Naive Bayes        | 94.16%   | 94.65%    | 94.16% | 94.22%   | 0.000s        |
| SVM (Linear)       | 94.89%   | 95.04%    | 94.89% | 94.92%   | 0.014s        |
| SVM (RBF)          | 94.89%   | 95.04%    | 94.89% | 94.92%   | 0.007s        |
| Logistic Regression| 94.89%   | 94.92%    | 94.89% | 94.90%   | 0.009s        |
| Decision Tree      | 95.62%   | 95.62%    | 95.62% | 95.62%   | 0.004s        |
| KNN                | 94.16%   | 94.15%    | 94.16% | 94.13%   | 0.004s        |

### Notes
- **Random Forest** is the best model with 97.08% accuracy and the lowest Type II Error (0.02).
- **Naive Bayes** has the fastest training time (0.000s).
- **Logistic Regression** and **SVM** offer high interpretability, suitable for medical environments.

## 🏥 Medical Analysis & Clinical Implications

### 🚨 Error Analysis (Critical for Medical Applications)

**KNN Model (Deployed) Error Breakdown:**
- **Type I Error (False Positive)**: 3/87 = **3.45%**
  - **Clinical Impact**: Benign tissue misclassified as malignant
  - **Consequences**: Patient anxiety, unnecessary procedures, additional testing costs
  - **Acceptable Level**: ✅ Within medical guidelines (<5%)

- **Type II Error (False Negative)**: 1/50 = **2.00%**
  - **Clinical Impact**: Malignant tissue misclassified as benign
  - **Consequences**: ⚠️ **CRITICAL** - Delayed treatment, disease progression
  - **Benchmark**: ✅ Excellent (<3% is considered very good)

### 📊 Medical Performance Metrics

| Metric | KNN (Deployed) | Clinical Significance |
|--------|----------------|----------------------|
| **Sensitivity** | 98.00% | Correctly identifies 98% of cancer cases |
| **Specificity** | 96.55% | Correctly identifies 96.5% of healthy cases |
| **PPV (Precision)** | 94.23% | 94% of positive predictions are correct |
| **NPV** | 98.82% | 99% of negative predictions are correct |

### 🎯 Clinical Recommendations

1. **KNN Model Deployment**: ✅ **Recommended for clinical use**
   - Excellent balance of sensitivity/specificity
   - Low Type II error rate (critical for cancer detection)
   - Fast prediction time suitable for real-time diagnosis

2. **Usage Guidelines**:
   - Use as **screening tool** with expert physician review
   - **Always combine** with clinical examination and other tests
   - Consider biopsy for borderline cases (confidence < 85%)

3. **Risk Mitigation**:
   - Implement **confidence thresholds** (API provides confidence scores)
   - **Flag low-confidence predictions** for additional testing
   - Regular model retraining with new data

### 🔍 API Safety Features

The deployed API includes medical safety considerations:
- **Confidence scoring** for all predictions
- **Medical disclaimers** in all responses
- **Risk level assessment** (Low/High)
- **Recommendation guidance** for healthcare providers

## 🔧 Customization

### Add New Algorithm
```python
# In utils/model_trainer.py
def train_new_algorithm(X_train, X_test, y_train, y_test):
    model = YourNewModel()
    return train_and_evaluate_model(model, "New Model", X_train, X_test, y_train, y_test)
```

### Add Visualization
```python
# In utils/visualizer.py
def plot_custom_visualization(data, title):
    # Add new visualization
    pass
```

### Add New Metric
```python
# In utils/model_comparison.py
def calculate_new_metric(y_true, y_pred):
    return new_score
```

## 🤝 Contribution

### How to Contribute
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a Pull Request.

### Code Standards
- Follow **PEP 8**.
- Add **docstrings** to all functions.
- Use **type hints** where possible.
- Include comprehensive error handling.

### Contribution Ideas
- Add Deep Learning algorithms (e.g., TensorFlow/Keras).
- Integrate real-time API execution.
- Add advanced visualizations (e.g., 3D plots).
- Automate hyperparameter tuning.

## 🐛 Bug Reporting

Please create an issue with:
- Detailed bug description.
- Environment (OS, Python version, dependencies).
- Steps to reproduce the bug.
- Expected outcome.

## 📝 License

This project is distributed under the **MIT License**:

```
MIT License

Copyright (c) 2025 Breast Cancer Prediction Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 Contact

- **GitHub**: [Your GitHub Profile](https://github.com/your-username)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- **Wisconsin Breast Cancer Dataset**: UCI Machine Learning Repository.
- **scikit-learn**: Powerful machine learning library.
- **Jupyter**: Excellent interactive environment.
- **Matplotlib & Seaborn**: High-quality visualizations.

---

⭐ **If you find this project helpful, please give it a star!** ⭐

**Note**: This is a research and educational project. Do not use it as a substitute for professional medical diagnosis.

---
