# Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Function to load a dataset
def load_dataset():
    """Loads the dataset and returns train/test splits."""
    return tf.keras.datasets.boston_housing.load_data()

# Function to analyze a dataset (statistical analysis)
def analyze_dataset(train_data, train_labels, test_data, test_labels):
    """
    Perform statistical analysis of the dataset, including:
    - Shape and data types
    - Missing values
    - Summary statistics

    Parameters:
        train_data (numpy.ndarray): Training feature set
        test_data (numpy.ndarray): Testing feature set
        train_labels (numpy.ndarray): Training labels
        test_labels (numpy.ndarray): Testing labels
    """

    # Print header for the function
    print("\n🎯 Dataset Analysis 🎯\n")

    # Convert to DataFrame for better analysis
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    train_labels_df = pd.DataFrame(train_labels, columns=[''])
    test_labels_df = pd.DataFrame(test_labels, columns=[''])

    # Dataset Shape and Data Types
    print("\n🔹 Dataset Shape & Data Types:\n")
    print(f"Train data shape: {train_data.shape}, Type: {train_data.dtype}")
    print(f"Test data shape: {test_data.shape}, Type: {test_data.dtype}")
    print(f"Train labels shape: {train_labels.shape}, Type: {train_labels.dtype}")
    print(f"Test labels shape: {test_labels.shape}, Type: {test_labels.dtype}")

    # Checking for Missing Values
    print("\n🔹 Missing Values:\n")
    print(f"Train data missing values: {np.isnan(train_data).sum()}")
    print(f"Test data missing values: {np.isnan(test_data).sum()}")
    print(f"Train labels missing values: {np.isnan(train_labels).sum()}")
    print(f"Test labels missing values: {np.isnan(test_labels).sum()}")

    # Summary Statistics (using DataFrame)
    print("\n🔹 Statistical Summary:\n")
    print("\nTrain Data Statistics:\n\n", train_df.describe())
    print("\nTest Data Statistics:\n\n", test_df.describe())
    print("\nTrain Labels Statistics:\n", train_labels_df.describe())
    print("\nTest Labels Statistics:\n", test_labels_df.describe())

# Function to preprocess a dataset (normalization, reshaping, etc.)
def preprocess_dataset(train_data, train_labels, test_data, test_labels):
    """
    Preprocesses data for regression models:
    - Reshapes labels
    - Prints pre-normalization min/max ranges
    - Applies MinMaxScaler normalization
    - Prints post-normalization min/max ranges
    - Converts data types to float32 for optimization

    Returns:
    - Scaled train_data, train_labels, test_data, test_labels
    """

    # Print header for the function
    print("\n🎯 Preprocessing Steps 🎯\n")

    # Reshape labels to ensure compatibility
    train_labels = np.reshape(train_labels, (-1, 1))
    test_labels = np.reshape(test_labels, (-1, 1))

    print("\n🔹 Shapes After Reshaping:")
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)

    # Check pre-normalization min/max values
    train_data_min, train_data_max = train_data.min(axis=0), train_data.max(axis=0)
    test_data_min, test_data_max = test_data.min(axis=0), test_data.max(axis=0)
    train_labels_min, train_labels_max = train_labels.min(axis=0), train_labels.max(axis=0)
    test_labels_min, test_labels_max = test_labels.min(axis=0), test_labels.max(axis=0)

    print("\n🔹 Pre-Normalization Data Ranges:")
    print("Train Data Min:", train_data_min, "\nTrain Data Max:", train_data_max)
    print("Test Data Min:", test_data_min, "\nTest Data Max:", test_data_max)

    # Fit the scaler on training data only
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data)

    # Transform both training and test data using the scaler
    train_data = min_max_scaler.transform(train_data)
    test_data = min_max_scaler.transform(test_data)

    # Check post-normalization min/max values
    train_min_post, train_max_post = train_data.min(axis=0), train_data.max(axis=0)
    test_min_post, test_max_post = test_data.min(axis=0), test_data.max(axis=0)

    print("\n🔹 Post-Normalization Data Ranges:")
    print("Post-Normalization Train Data Min:", train_min_post, "\nPost-Normalization Train Data Max:", train_max_post)
    print("Post-Normalization Test Data Min:", test_min_post, "\nPost-Normalization Test Data Max:", test_max_post)

    # Print min/max values for labels
    print("\n🔹 (Optional) Label Ranges:")
    print("Train Labels Min:", train_labels_min, "\nTrain Labels Max:", train_labels_max)
    print("Test Labels Min:", test_labels_min, "\nTest Labels Max:", test_labels_max)

    # Convert dataset values to float32 for optimization
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    print("\n🔹 Data Types After Conversion:")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    return train_data, train_labels, test_data, test_labels

# Print confirmation message
print("\n✅ data.py successfully executed\n")