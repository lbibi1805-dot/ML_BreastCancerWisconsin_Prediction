# 🩺 Breast Cancer Prediction - Machine Learning Comparison

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dự án này so sánh toàn diện 7 thuật toán machine learning để dự đoán ung thư vú, sử dụng **Wisconsin Breast Cancer Dataset**. Với trọng tâm là ứng dụng y tế, dự án cung cấp phân tích lỗi Type I/Type II và CAP Analysis để đảm bảo độ an toàn và hiệu quả trong chẩn đoán.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Dataset Structure Explanation](#dataset-structure-explanation)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Hiệu suất thuật toán](#hiệu-suất-thuật-toán)
- [Phân tích y tế](#phân-tích-y-tế)
- [Tùy chỉnh](#tùy-chỉnh)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)
- [Liên hệ](#liên-hệ)

## 🎯 Tổng quan

Dự án này triển khai và so sánh các thuật toán machine learning để phân loại ung thư vú thành **lành tính (benign)** hoặc **ác tính (malignant)**. Được thiết kế với kiến trúc modular, dự án nhấn mạnh vào:

- **Hiệu suất tối ưu**: So sánh 7 thuật toán với các metric chuyên sâu.
- **An toàn y tế**: Phân tích lỗi Type I (False Positive) và Type II (False Negative).
- **Triển khai lâm sàng**: Đánh giá khả năng áp dụng thực tế trong y tế.

Dataset được sử dụng là **Wisconsin Breast Cancer Dataset** từ UCI Machine Learning Repository, với 9 đặc trưng và 2 lớp (benign: 2, malignant: 4).

## ✨ Tính năng

### 🔬 Machine Learning
- **7 thuật toán**: Logistic Regression, KNN, SVM (Linear & RBF), Decision Tree, Random Forest, Naive Bayes.
- **Preprocessing thống nhất**: Feature scaling và xử lý dữ liệu đồng bộ.
- **Tối ưu hóa hyperparameters**: Tìm K tối ưu cho KNN, so sánh kernel cho SVM.
- **Cross-validation**: Đánh giá hiệu suất ổn định với 10-fold CV.

### 📊 Visualization
- **Confusion Matrix**: Hiển thị chi tiết lỗi phân loại.
- **Decision Boundary**: Biên quyết định 2D cho từng thuật toán.
- **Feature Importance**: Phân tích mức độ quan trọng của các đặc trưng.
- **CAP Curves**: Đánh giá khả năng phân biệt trong y tế.
- **Error Analysis Plots**: So sánh Type I/Type II errors.

### 🏥 Medical Analysis
- **Type I/II Error Analysis**: Đánh giá lỗi dương tính giả và âm tính giả.
- **CAP Analysis**: Cumulative Accuracy Profile cho đánh giá y tế.
- **Clinical Recommendations**: Hướng dẫn triển khai dựa trên an toàn và hiệu quả.

### 💾 Model Persistence
- **Lưu/tải model**: Lưu models với metadata (accuracy, hyperparameters, timestamp).
- **Batch processing**: Quản lý nhiều models cùng lúc.
- **Production-ready**: Hàm dự đoán dễ tích hợp vào hệ thống y tế.

## 📊 Dataset Structure Explanation

**🎯 Dependent Variable (Target Variable):**
- **`Class`**: Phân loại ung thư vú
  - **2**: Benign (Lành tính) - Không có ung thư
  - **4**: Malignant (Ác tính) - Có ung thư

**🔬 Independent Variables (Features):** 9 đặc trưng y tế từ mẫu tế bào

1. **`clump_thickness`**: Độ dày cụm tế bào (1-10)
   - Giá trị cao → Nghi ngờ ác tính
   
2. **`uniform_cell_size`**: Tính đồng đều kích thước tế bào (1-10)
   - Tế bào ác tính thường có kích thước không đồng đều
   
3. **`uniform_cell_shape`**: Tính đồng đều hình dạng tế bào (1-10)
   - Tế bào ác tính thường có hình dạng bất thường
   
4. **`marginal_adhesion`**: Độ bám dính biên tế bào (1-10)
   - Tế bào ác tính có xu hướng mất khả năng bám dính
   
5. **`single_epithelial_cell_size`**: Kích thước tế bào biểu mô đơn (1-10)
   - Liên quan đến sự phát triển bất thường của tế bào
   
6. **`bare_nuclei`**: Nhân trần (không có tế bào chất bao quanh) (1-10)
   - Đặc trưng thường thấy ở ung thư ác tính
   
7. **`bland_chromatin`**: Cấu trúc nhiễm sắc thể (1-10)
   - Tế bào ác tính có cấu trúc nhiễm sắc thể bất thường
   
8. **`normal_nucleoli`**: Nhân con bình thường (1-10)
   - Tế bào ác tính có nhân con to và nổi bật
   
9. **`mitoses`**: Quá trình phân bào (1-10)
   - Tế bào ác tính có tỷ lệ phân bào cao

**📈 Tầm Quan Trọng trong Machine Learning:**
- **Features (X)**: 9 đặc trưng y tế → Dữ liệu đầu vào để dự đoán
- **Target (y)**: Phân loại ung thư → Kết quả cần dự đoán
- **Mục tiêu**: Học từ features để dự đoán chính xác target

## 📁 Cấu trúc dự án

```
ML_BreastCancerWisconsin_Prediction/
│
├── 📊 Dataset/
│   ├── breast_cancer_wisconsin.csv      # Dataset chính
│   ├── breast_cancer.csv                # Dataset phụ
│   ├── Source.txt                       # Thông tin nguồn
│   └── raw_data/                        # Dữ liệu gốc
│       ├── breast-cancer-wisconsin.data
│       ├── breast-cancer-wisconsin.names
│       ├── wdbc.data
│       ├── wdbc.names
│       ├── wpbc.data
│       └── wpbc.names
│
├── 💻 Codes/
│   ├── 📓 ml_models_comparison.ipynb    # Notebook chính
│   ├── 🛠️ utils/                        # Package tiện ích
│   │   ├── __init__.py                  # Khởi tạo package
│   │   ├── data_processor.py            # Xử lý dữ liệu
│   │   ├── model_trainer.py             # Huấn luyện model
│   │   ├── visualizer.py                # Visualization
│   │   ├── model_persistence.py         # Lưu/tải model
│   │   ├── model_comparison.py          # So sánh model
│   │   ├── medical_error_analysis.py    # Phân tích lỗi y tế
│   │   └── cap_analysis.py              # CAP analysis
│   ├── DataCrawler.py                   # Script thu thập dữ liệu
│   └── logistic_regression.ipynb        # Notebook riêng cho Logistic Regression
│
├── 🤖 Models/                           # Thư mục lưu models
│   ├── Logistic_Regression.pkl
│   ├── KNN.pkl
│   ├── SVM_Linear.pkl
│   ├── SVM_RBF.pkl
│   ├── Decision_Tree.pkl
│   ├── Random_Forest.pkl
│   ├── Naive_Bayes.pkl
│   └── metadata/                        # Metadata của models
│
├── 📜 requirements.txt                  # Dependencies
└── 📖 README.md                         # File này
```

## 🚀 Cài đặt

### Yêu cầu
- Python 3.8+
- Jupyter Notebook/JupyterLab
- Git (tùy chọn)

### Bước 1: Clone repository
```bash
git clone https://github.com/your-username/ML_BreastCancerWisconsin_Prediction.git
cd ML_BreastCancerWisconsin_Prediction
```

### Bước 2: Tạo virtual environment
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate
# Kích hoạt (macOS/Linux)
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy Jupyter Notebook
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

## 💡 Sử dụng

### Quick Start
1. Mở file `ml_models_comparison.ipynb`.
2. Chạy tất cả cells để xem kết quả so sánh và visualizations.
3. Xem output để chọn model tốt nhất (Random Forest được khuyến nghị).

### Ví dụ sử dụng
```python
# Import modules
from utils import *

# Load và preprocess dữ liệu
dataset, feature_names = load_and_explore_data("../Dataset/breast_cancer_wisconsin.csv")
X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset, feature_names)

# Train và đánh giá model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
results = train_and_evaluate_model(model, "Random Forest", X_train, X_test, y_train, y_test)

# Visualize
plot_confusion_matrix(results)
plot_decision_boundary(model, "Random Forest", X_train, y_train, feature_names)

# Lưu model
save_model(model, results, "Random_Forest", save_dir="../Models")
```

### Tải và sử dụng model đã lưu
```python
# Tải model
loaded_model, metadata = load_model_by_name("Random_Forest", save_dir="../Models")
prediction = loaded_model.predict(new_data)
```

## 🤖 Hiệu suất thuật toán

| Thuật toán         | Accuracy | Precision | Recall | F1-Score | Training Time |
|--------------------|----------|-----------|--------|----------|---------------|
| Random Forest      | 97.08%   | 97.15%    | 97.08% | 97.09%   | 0.029s        |
| Naive Bayes        | 94.16%   | 94.65%    | 94.16% | 94.22%   | 0.000s        |
| SVM (Linear)       | 94.89%   | 95.04%    | 94.89% | 94.92%   | 0.014s        |
| SVM (RBF)          | 94.89%   | 95.04%    | 94.89% | 94.92%   | 0.007s        |
| Logistic Regression| 94.89%   | 94.92%    | 94.89% | 94.90%   | 0.009s        |
| Decision Tree      | 95.62%   | 95.62%    | 95.62% | 95.62%   | 0.004s        |
| KNN                | 94.16%   | 94.15%    | 94.16% | 94.13%   | 0.004s        |

### Ghi chú
- **Random Forest** là model tốt nhất với accuracy 97.08% và Type II Error thấp nhất (0.02).
- **Naive Bayes** có thời gian huấn luyện nhanh nhất (0.000s).
- **Logistic Regression** và **SVM** có tính giải thích cao, phù hợp với môi trường y tế.

## 🏥 Phân tích y tế

### Type I vs Type II Errors
- **Type I (False Positive)**: Chẩn đoán nhầm lành tính thành ác tính.
  - **Hậu quả**: Gây lo lắng, cần xét nghiệm thêm.
  - **Tỷ lệ thấp nhất**: Random Forest (0.015).
- **Type II (False Negative)**: Chẩn đoán nhầm ác tính thành lành tính.
  - **Hậu quả**: Bỏ sót ung thư, rất nguy hiểm.
  - **Tỷ lệ thấp nhất**: Random Forest (0.020).

### CAP Analysis
- **Accuracy Ratio**: Tất cả models đạt ~1.0 (Excellent).
- **CAP AUC**: Random Forest cao nhất (81.285).
- **Clinical Implication**: Random Forest có khả năng phân biệt tốt nhất.

### Khuyến nghị lâm sàng
- **Random Forest**: Lựa chọn an toàn nhất với Type II Error thấp nhất và CAP AUC cao nhất.
- **Logistic Regression**: Phù hợp khi cần giải thích rõ ràng cho bác sĩ.
- **SVM**: Ổn định với dữ liệu mới, phù hợp cho triển khai lâu dài.

## 🔧 Tùy chỉnh

### Thêm thuật toán mới
```python
# Trong utils/model_trainer.py
def train_new_algorithm(X_train, X_test, y_train, y_test):
    model = YourNewModel()
    return train_and_evaluate_model(model, "New Model", X_train, X_test, y_train, y_test)
```

### Thêm visualization
```python
# Trong utils/visualizer.py
def plot_custom_visualization(data, title):
    # Thêm visualization mới
    pass
```

### Thêm metric mới
```python
# Trong utils/model_comparison.py
def calculate_new_metric(y_true, y_pred):
    return new_score
```

## 🤝 Đóng góp

### Cách đóng góp
1. Fork repository.
2. Tạo branch mới (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to branch (`git push origin feature/YourFeature`).
5. Tạo Pull Request.

### Tiêu chuẩn code
- Tuân theo **PEP 8**.
- Thêm **docstrings** cho tất cả functions.
- Sử dụng **type hints** khi có thể.
- Xử lý lỗi đầy đủ.

### Ý tưởng đóng góp
- Thêm thuật toán Deep Learning (e.g., TensorFlow/Keras).
- Tích hợp API thực thi thời gian thực.
- Thêm visualizations nâng cao (e.g., 3D plots).
- Tự động hóa hyperparameter tuning.

## 🐛 Báo lỗi

Vui lòng tạo issue với:
- Mô tả lỗi chi tiết.
- Môi trường (OS, Python version, dependencies).
- Cách tái hiện lỗi.
- Kết quả mong đợi.

## 📝 Giấy phép

Dự án được phân phối dưới **MIT License**:

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

## 📞 Liên hệ

- **GitHub**: [Your GitHub Profile](https://github.com/your-username)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/your-profile)

## 🙏 Ghi nhận

- **Wisconsin Breast Cancer Dataset**: UCI Machine Learning Repository.
- **scikit-learn**: Thư viện machine learning mạnh mẽ.
- **Jupyter**: Môi trường tương tác tuyệt vời.
- **Matplotlib & Seaborn**: Visualization chất lượng cao.

---

⭐ **Nếu dự án hữu ích, hãy cho một star!** ⭐

**Lưu ý**: Đây là dự án nghiên cứu và giáo dục. Không sử dụng để thay thế chẩn đoán y tế chuyên nghiệp.

---