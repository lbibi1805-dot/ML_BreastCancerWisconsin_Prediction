"""
Data Processing Module
Handles data loading, preprocessing, and feature scaling
"""

import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_explore_data(file_path):
    """
    Load dataset and perform initial exploration
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV dataset file
        
    Returns:
    --------
    dataset : pandas.DataFrame
        Loaded dataset
    feature_names : list
        List of feature names (excluding sample code and target)
    """
    # Load the dataset
    dataset = pd.read_csv(file_path)

    print("Dataset Information:")
    print("=" * 40)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of features: {dataset.shape[1] - 2}")  # Excluding sample code and target
    print(f"Number of samples: {dataset.shape[0]}")

    # Display basic information
    print("\nDataset Info:")
    print("=" * 20)
    dataset.info()

    print("\nFirst 5 rows:")
    print("=" * 20)
    print(dataset.head())

    print("\nDataset Description:")
    print("=" * 25)
    print(dataset.describe())

    print("\nNull Values Check:")
    print("=" * 25)
    null_counts = dataset.isnull().sum()
    print(null_counts)
    print(f"Total null values: {null_counts.sum()}")

    # Target variable distribution
    print("\nTarget Variable Distribution:")
    print("=" * 35)
    target_counts = dataset.iloc[:, -1].value_counts()
    print(target_counts)
    print(f"Class distribution: {target_counts.values}")
    print(f"Balance ratio: {target_counts.min() / target_counts.max():.3f}")

    # Feature names
    feature_names = dataset.columns[1:-1].tolist()  # Exclude sample code and target
    print(f"\nFeature names ({len(feature_names)} features):")
    print("=" * 40)
    for i, name in enumerate(feature_names, 1):
        print(f"{i:2d}. {name}")

    print("\n" + "=" * 60)
    print("Data loading and exploration completed successfully!")
    print("=" * 60)
    
    return dataset, feature_names


def preprocess_data(dataset, feature_names, test_size=0.2, random_state=0):
    """
    Complete data preprocessing pipeline (giống y hệt code từ các file con)
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        Input dataset
    feature_names : list
        List of feature names
    test_size : float
        Test set proportion
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Preprocessed training and testing data
    scaler : StandardScaler
        Fitted scaler object
    """
    # Process null values (giống y hệt code từ các file con)
    threshold = 0.05

    if dataset.isna().sum().sum() / dataset.size < threshold:
        dataset = dataset.dropna()
    else:
        for col in dataset.columns:
            if dataset[col].dtype in ['float64', 'int64']:
                # Điền NaN bằng mean cho dữ liệu số
                dataset[col] = dataset[col].fillna(dataset[col].mean())
            else:
                # Điền NaN bằng giá trị phổ biến nhất cho dữ liệu dạng object/categorical
                dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

    print("Number of null data after processing:")
    print("__________________________")
    pprint(pd.isnull(dataset).sum())

    # Declare features and dependent variables (giống y hệt code từ các file con)
    print("\nDeclaring features and dependent variables...")
    print("=" * 50)
    print("On the features, remove the 'Sample code number' because it is not relevant to the prediction")

    X = dataset.iloc[:,1:-1].values
    y = dataset.iloc[:, -1].values

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # Splitting the dataset into Training set and Test set (giống y hệt code từ các file con)
    print("\nSplitting the dataset into Training set and Test set...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("DataSet Splitting:")
    print("_______________________________")
    print("X_train: ", np.size(X_train))
    print("X_test: ", np.size(X_test))
    print("y_train:", np.size(y_train))
    print("y_test", np.size(y_test))

    # Feature Scaling (giống y hệt code từ các file con)
    print("\nFeature Scaling...")
    print("=" * 20)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Feature Scaling Applied Successfully!")
    print("_____________________________________")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("\nTraining set - First 5 samples after scaling:")
    pprint(X_train[:5])
    print("\nTest set - First 5 samples after scaling:")
    pprint(X_test[:5])

    print(f"\n" + "=" * 60)
    print("Data preprocessing completed successfully!")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, scaler
