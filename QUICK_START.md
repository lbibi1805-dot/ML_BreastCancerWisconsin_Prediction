# 🚀 Quick Start Guide

## 🌐 Use Live API (Recommended)

**Production URL**: https://api-deploy-ml-breastcancer-wisconsin.onrender.com

```bash
# Test prediction
curl -X POST https://api-deploy-ml-breastcancer-wisconsin.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2, 1, 1, 1, 2, 1, 2, 1, 1]}'
```

**✅ Result**: `{"prediction": 2, "prediction_label": "Benign", "confidence": 0.95}`

## 💻 Local Development

```bash
# Clone and setup
git clone [repository]
cd ML_BreastCancerWisconsin_Prediction
pip install -r requirements.txt

# Quick test KNN model
cd prediction_tools
python test_knn.py

# Train models (Jupyter)
cd Codes
jupyter notebook ml_models_comparison.ipynb

# Run local API
cd api_server
python app.py
```

## 📱 Frontend Integration

```javascript
const predictCancer = async (features) => {
  const response = await fetch('https://api-deploy-ml-breastcancer-wisconsin.onrender.com/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features })
  });
  return await response.json();
};

// Example usage
const features = [2, 1, 1, 1, 2, 1, 2, 1, 1]; // 9 medical features (1-10 scale)
const result = await predictCancer(features);
console.log(result); // {prediction: 2, prediction_label: "Benign", confidence: 0.95}
```

## 🔬 Input Features (1-10 scale)

1. **Clump Thickness**
2. **Uniform Cell Size**  
3. **Uniform Cell Shape**
4. **Marginal Adhesion**
5. **Single Epithelial Cell Size**
6. **Bare Nuclei**
7. **Bland Chromatin**
8. **Normal Nucleoli**
9. **Mitoses**

## 📊 Output

- **prediction**: `2` (Benign) or `4` (Malignant)
- **prediction_label**: "Benign" or "Malignant"
- **confidence**: 0.0-1.0 (model confidence)

---
📖 **Full Documentation**: [README.md](README.md) | **API Docs**: [API Repository](../API_Deploy_ML_BreastCancer_Wisconsin/README.md)
