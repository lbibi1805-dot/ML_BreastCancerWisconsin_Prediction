# 🔧 Prediction Testing Tools

**Local testing utilities for trained models**

## 🚀 Quick Usage

### Test KNN Model (Deployed Version)
```bash
cd prediction_tools
python test_knn.py
```

### Interactive KNN App
```bash
python knn_cancer_app.py
```

### Multi-Model Testing
```bash
python single_prediction_test.py
```

## 📊 Sample Test Cases

**Benign case:** `[2, 1, 1, 1, 2, 1, 2, 1, 1]`
**Malignant case:** `[8, 7, 8, 7, 6, 9, 7, 8, 3]`

## ⚠️ Prerequisites

1. Train models first: Run `Codes/ml_models_comparison.ipynb`
2. Ensure Models/ directory contains .joblib files
3. Run from project root directory

---

**💡 For API usage, see main README.md or use live API:**
https://api-deploy-ml-breastcancer-wisconsin.onrender.com
**KNN Cancer Prediction Application**
- Complete application using only KNN (k=3) model
- Medical-grade interface and interpretation
- Interactive mode for real patient data
- Focused on the chosen KNN algorithm

**Usage:**
```bash
cd prediction_tools
python knn_cancer_app.py
```

### `quick_test.py` 
**Quick verification script**
- Loads Random Forest model directly from `../Models/` folder
- Tests with 2 sample cases (benign and malignant)
- Shows basic prediction output
- No external dependencies beyond numpy

**Usage:**
```bash
cd prediction_tools
python quick_test.py
```

### `single_prediction_test.py`
**Multi-model testing system**
- Loads all available models from `../Models/` folder
- Tests with 3 sample patients (low/moderate/high risk)
- Consensus predictions from multiple models
- Interactive mode for custom patient data

**Usage:**
```bash
cd prediction_tools
python single_prediction_test.py
```

## 🔧 How It Works

### Model Loading
- **Direct file access**: Scripts load `.joblib` and `_metadata.json` files directly
- **No utils dependency**: Uses `joblib` and `json` libraries instead of custom utils
- **Automatic discovery**: Finds model files based on naming pattern
- **Timestamp handling**: Uses most recent model files if multiple exist

### File Structure Expected
```
../Models/
├── Random Forest_20250720_110419.joblib
├── Random Forest_20250720_110419_metadata.json
├── Logistic Regression_20250720_110419.joblib
├── Logistic Regression_20250720_110419_metadata.json
└── ...
```

### Patient Data Format
9 features (scale 1-10):
1. `clump_thickness`: Cell clump thickness
2. `uniform_cell_size`: Cell size uniformity
3. `uniform_cell_shape`: Cell shape uniformity
4. `marginal_adhesion`: Cell adhesion
5. `single_epithelial_cell_size`: Epithelial cell size
6. `bare_nuclei`: Bare nuclei presence
7. `bland_chromatin`: Chromatin structure
8. `normal_nucleoli`: Nucleoli normality
9. `mitoses`: Mitosis frequency

## 📊 Sample Test Cases

### Low Risk (Expected: Benign)
```python
[2, 1, 1, 1, 2, 1, 2, 1, 1]
```

### High Risk (Expected: Malignant)
```python
[8, 7, 8, 7, 6, 9, 7, 8, 3]
```

## 🚀 Running Tests

1. **Ensure models are trained:**
   ```bash
   cd ../Codes
   # Run ml_models_comparison.ipynb notebook
   ```

2. **Run quick test:**
   ```bash
   cd prediction_tools
   python quick_test.py
   ```

3. **Run comprehensive test:**
   ```bash
   cd prediction_tools
   python single_prediction_test.py
   ```

## ⚠️ Important Notes

- **No scaling needed**: Models handle preprocessing internally
- **Direct model loading**: No dependency on utils package
- **Raw feature values**: Input values 1-10 as specified in dataset
- **Medical interpretation**: 2=Benign, 4=Malignant
- **For research only**: Not for actual medical diagnosis

## 🔍 Troubleshooting

### "Model not found"
- Check that `../Models/` directory exists
- Ensure `.joblib` and `_metadata.json` files are present
- Run notebook to train models first

### "Import errors"  
- Scripts only use standard libraries: `numpy`, `joblib`, `json`
- No custom utils dependencies

### "Prediction errors"
- Check input data format (9 features, values 1-10)
- Ensure model files are not corrupted
