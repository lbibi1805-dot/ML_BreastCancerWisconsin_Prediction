from ucimlrepo import fetch_ucirepo 
import pandas as pd

# Fetch dataset
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 

# Lấy đặc trưng (X) và nhãn (y)
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

# Gộp lại thành 1 DataFrame
df = pd.concat([X, y], axis=1)

# Lưu thành file CSV (có encoding để đọc tốt với Excel)
df.to_csv('../Dataset/breast_cancer_wisconsin.csv', index=False, encoding='utf-8-sig')

