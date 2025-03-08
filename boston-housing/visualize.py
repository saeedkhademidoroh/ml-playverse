# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to visualize model training history
def visualize_model_history(model_history):
    """
    Plots the training and validation metrics of a Keras model.

    Parameters:
    model_history (History): The History object returned by the fit method of a Keras model.
    """

    # Print header for the function
    print("\n🎯 Training History Visualization 🎯\n")

    # Convert the history.history dictionary to a DataFrame
    history_df = pd.DataFrame(model_history.history)

    # Rename columns for better readability
    history_df.rename(columns={
        'loss': 'Training Loss',
        'val_loss': 'Validation Loss'
    }, inplace=True)

    # Plot the DataFrame
    history_df.plot(figsize=(10, 6))
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)

    # Display the plot
    plt.show()

# Function to visualize a dataset (plotting)
def visualize_dataset(train_data, train_labels, test_data, test_labels):
    """
    Visualize the dataset by plotting:
    - Feature distributions
    - Correlation heatmap
    - Outlier detection (boxplots)
    - Label distribution

    Parameters:
        train_data (numpy.ndarray): Training feature set
        test_data (numpy.ndarray): Testing feature set
        train_labels (numpy.ndarray): Training labels
        test_labels (numpy.ndarray): Testing labels
    """

    # Print header for the function
    print("\n🎯 Dataset Visualization 🎯\n")

    # Feature Distributions
    num_features = train_data.shape[1]
    plt.figure(figsize=(15, num_features * 2))
    for i in range(num_features):
        plt.subplot((num_features // 3) + 1, 3, i + 1)
        sns.histplot(train_data[:, i], kde=True, bins=30, color="blue", label="Train")
        sns.histplot(test_data[:, i], kde=True, bins=30, color="orange", label="Test")
        plt.xlabel(f"Feature {i}")
        plt.ylabel("Count")
        plt.legend()
    plt.suptitle("Feature Distributions (Train vs. Test)\n\n")
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = pd.DataFrame(train_data).corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap\n")
    plt.show()

    # Outlier Detection (Boxplots)
    plt.figure(figsize=(15, num_features * 2))
    for i in range(num_features):
        plt.subplot((num_features // 3) + 1, 3, i + 1)
        sns.boxplot(x=train_data[:, i], color="red")
        plt.xlabel(f"Feature {i}")
    plt.suptitle("Feature Outlier Detection (Boxplots)\n\n")
    plt.tight_layout()
    plt.show()

    # Label Distribution
    plt.figure(figsize=(10, 4))
    sns.histplot(train_labels, kde=True, bins=30, color="blue", label="Train Labels")
    sns.histplot(test_labels, kde=True, bins=30, color="orange", label="Test Labels")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Label Distribution (Train vs. Test)\n")
    plt.show()

# Print confirmation message
print("\n✅ visualize.py successfully executed\n")