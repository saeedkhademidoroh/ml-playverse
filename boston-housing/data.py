import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def load_dataset():
    """Loads the Boston housing dataset and returns train/test splits."""
    return tf.keras.datasets.boston_housing.load_data()

def analyze_dataset(train_data, train_labels, test_data, test_labels):
    """Performs statistical analysis of the dataset."""
    print("\n🎯 Dataset Analysis 🎯\n")

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print("\n🔹 Shape & Types:")
    print(f"Train: {train_data.shape}, Type: {train_data.dtype}")
    print(f"Test: {test_data.shape}, Type: {test_data.dtype}")

    print("\n🔹 Missing Values:")
    print(f"Train Missing: {np.isnan(train_data).sum()}")
    print(f"Test Missing: {np.isnan(test_data).sum()}")

    print("\n🔹 Statistical Summary:")
    print(train_df.describe())

def preprocess_dataset(train_data, train_labels, test_data, test_labels):
    """Applies MinMax scaling and reshapes labels."""
    print("\n🎯 Preprocessing Dataset 🎯\n")

    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data.astype(np.float32), train_labels.astype(np.float32), test_data.astype(np.float32), test_labels.astype(np.float32)
