# 🩺 Breast Cancer Prediction - ML Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-Live-brightgreen.svg)](https://api-deploy-ml-breastcancer-wisconsin.onrender.com)

Comprehensive machine learning project for breast cancer prediction using the Wisconsin Breast Cancer Dataset. Features 7 ML algorithms comparison, medical analysis, and **live API deployment**.

## 🚀 Live API Usage

**🌐 Production API**: https://api-deploy-ml-breastcancer-wisconsin.onrender.com

```bash
# Quick test
curl -X POST https://api-deploy-ml-breastcancer-wisconsin.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2, 1, 1, 1, 2, 1, 2, 1, 1]}'
```

**📱 Frontend Integration:**
```javascript
const predictCancer = async (features) => {
  const response = await fetch('https://api-deploy-ml-breastcancer-wisconsin.onrender.com/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features })
  });
  return await response.json();
};
```

**📖 Complete API Documentation**: [API Repository](../API_Deploy_ML_BreastCancer_Wisconsin/README.md)

## 🎯 Model Performance Summary

| Algorithm | Test Accuracy | F1-Score | Status |
|-----------|---------------|----------|---------|
| **🥇 KNN (k=3)** | **97.08%** | **97.09%** | ✅ **DEPLOYED** |
| 🥈 Random Forest | 97.08% | 96.77% | ✅ Available |
| 🥉 Logistic Regression | 96.35% | 95.83% | ✅ Available |
| SVM (RBF) | 96.35% | 95.83% | ✅ Available |
| SVM (Linear) | 95.62% | 94.59% | ✅ Available |
| Decision Tree | 95.62% | 94.59% | ✅ Available |
| Naive Bayes | 94.89% | 93.88% | ✅ Available |

## 📊 Dataset Information

**Wisconsin Breast Cancer Dataset**
- **Samples**: 699 patients
- **Features**: 9 medical characteristics (1-10 scale)
- **Classes**: Benign (2) vs Malignant (4)
- **Source**: UCI Machine Learning Repository

**🔬 Features:**
1. `clump_thickness` - Cell clump thickness
2. `uniform_cell_size` - Cell size uniformity  
3. `uniform_cell_shape` - Cell shape uniformity
4. `marginal_adhesion` - Cell adhesion quality
5. `single_epithelial_cell_size` - Epithelial cell size
6. `bare_nuclei` - Bare nuclei presence
7. `bland_chromatin` - Chromatin structure
8. `normal_nucleoli` - Nucleoli normality
9. `mitoses` - Mitosis frequency

## 💻 Local Development

### Quick Start (Jupyter Notebooks)
```bash
git clone [repository]
cd ML_BreastCancerWisconsin_Prediction
pip install -r requirements.txt

# Train all models
cd Codes
jupyter notebook ml_models_comparison.ipynb
```

### Local API Server
```bash
cd api_server
python app.py
# Server: http://localhost:5000
```

### Testing Tools
```bash
cd prediction_tools

# Test KNN model
python test_knn.py

# Interactive prediction app
python knn_cancer_app.py
```

## 🏥 Medical Analysis

### 📊 KNN Model (Deployed) Performance
- **Sensitivity**: 98.00% (correctly identifies cancer)
- **Specificity**: 96.55% (correctly identifies healthy)
- **Type I Error**: 3.45% (false positive - acceptable)
- **Type II Error**: 2.00% (false negative - excellent)

### 🎯 Clinical Significance
- **Type I**: Benign → Malignant (causes anxiety, extra tests)
- **Type II**: Malignant → Benign (⚠️ **CRITICAL** - missed cancer)
- **KNN Performance**: Excellent balance for medical screening

### ⚠️ Medical Disclaimer
- **For research/educational purposes only**
- **Not for actual medical diagnosis**
- Always consult healthcare professionals
- Use as screening tool with expert review

## 📁 Project Structure

```
ML_BreastCancerWisconsin_Prediction/
├── 📓 Codes/
│   ├── ml_models_comparison.ipynb     # Complete model training & comparison
│   ├── knn_neighbours.ipynb           # KNN implementation (deployed)
│   ├── [other model notebooks...]
│   ├── streamlit_app.py              # Web interface
│   └── utils/                        # Reusable ML modules
│
├── 📊 Dataset/
│   └── breast_cancer_wisconsin.csv   # Main dataset
│
├── 🤖 Models/                         # Trained models (.joblib + metadata)
│   ├── KNN_20250720_110419.joblib   # ✅ DEPLOYED MODEL
│   └── [other trained models...]
│
└── 🔧 prediction_tools/               # Local testing scripts
    ├── test_knn.py                   # Quick KNN test
    ├── knn_cancer_app.py            # Interactive app
    └── single_prediction_test.py     # Multi-model testing
```

## 🔗 Related Resources

### 🌐 API Deployment
- **API Repository**: [API_Deploy_ML_BreastCancer_Wisconsin](../API_Deploy_ML_BreastCancer_Wisconsin)
- **Live URL**: https://api-deploy-ml-breastcancer-wisconsin.onrender.com
- **API Docs**: Complete documentation with examples

### 📚 Documentation
- **API Usage**: See API repository README
- **Local Setup**: This README
- **Testing Guide**: `prediction_tools/README.md`

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** Pull Request

## 📈 Future Enhancements

- [ ] **Deep Learning**: CNN/RNN implementations
- [ ] **Feature Engineering**: Advanced selection techniques
- [ ] **Model Interpretability**: SHAP/LIME integration
- [ ] **Mobile App**: React Native implementation
- [ ] **Real-time Monitoring**: Performance tracking

## 📝 License

This project is distributed under the **MIT License**.

---

**🏥 Medical Disclaimer**: This is a research tool, not a replacement for professional medical diagnosis. Always consult healthcare professionals for medical decisions.
