# Third-party imports
import numpy as np  # Efficient array operations
import pandas as pd  # Data structures for handling structured data
import tensorflow as tf  # Core machine learning framework
from sklearn.preprocessing import MinMaxScaler  # Scales data to a range [0, 1]
from keras.api.utils import to_categorical # One-hot encoding

# Function to load a dataset
def load_dataset():
    """Loads the dataset and returns train/test splits."""
    return tf.keras.datasets.mnist.load_data()

# Function to analyze a dataset (statistical analysis)
def analyze_dataset(train_data, train_labels, test_data, test_labels):
    """
    Perform statistical analysis of the dataset, including:
    - Shape and data types
    - Missing values
    - Summary statistics (for 2D datasets)

    Supports both 2D and 3D datasets (e.g., Boston Housing and MNIST).

    Parameters:
        train_data (numpy.ndarray): Training feature set
        test_data (numpy.ndarray): Testing feature set
        train_labels (numpy.ndarray): Training labels
        test_labels (numpy.ndarray): Testing labels
    """

    # Print header for the function
    print("\n🎯 Dataset Analysis 🎯")

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

    # Handle different dataset types (2D vs 3D)
    if train_data.ndim == 2:  # 2D Data (Boston Housing, Tabular Data)
        # Convert to DataFrame for better analysis
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        train_labels_df = pd.DataFrame(train_labels, columns=['Label'])
        test_labels_df = pd.DataFrame(test_labels, columns=['Label'])

        # Summary Statistics (for 2D datasets)
        print("\n🔹 Statistical Summary:\n")
        print("Train Data", train_df.describe())
        print("\nTest Data", test_df.describe())
        print("\nTrain Labels", train_labels_df.describe())
        print("\nTest Labels", test_labels_df.describe())

    elif train_data.ndim == 3:  # 3D Data (MNIST Images)
        print("\n🔹 Image Data - Cannot display statistics in DataFrame format.")
        print(f"Each image has shape: {train_data.shape[1:]} (Height x Width)")
        print(f"Pixel value range: {train_data.min()} to {train_data.max()}")

        # Flatten one example image to show basic stats
        example_image = train_data[0].flatten()
        print("\n🔹 Sample Image Statistics:")
        print(f"Min Pixel Value: {example_image.min()}, Max Pixel Value: {example_image.max()}")
        print(f"Mean Pixel Value: {example_image.mean()}, Std Dev: {example_image.std()}")

    else:
        print("⚠️ Unsupported data shape. Only 2D and 3D datasets are supported.")

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
    print("\n🎯 Preprocessing Steps 🎯")

    # Print data types before preprocessing
    print("\n🔹 Data Types Before Preprocessing:\n")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    # Print shapes before preprocessing
    print("\n🔹 Data Shapes Before Preprocessing:\n")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)
    
    # Reshape data to ensure compatibility
    train_data = np.reshape(train_data, (-1, 28 * 28))
    test_data = np.reshape(test_data, (-1, 28 * 28))

    # Normalize the data with dividing by 255 and convert to float32
    train_data = (train_data / 255.0).astype(np.float32)
    test_data = (test_data / 255.0).astype(np.float32)

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Print data types after preprocessing
    print("\n🔹 Data Types After Preprocessing:\n")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    # Print shapes after preprocessing
    print("\n🔹 Data Shapes After Preprocessing:\n")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)

    return train_data, train_labels, test_data, test_labels

# Print confirmation message
print("\n✅ data.py successfully executed")