# Third-party imports
import pandas as pd  # Data manipulation with pandas
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Enhanced data visualization based on matplotlib

# Function to visualize dataset
def visualize_dataset(train_data, train_labels, test_data, test_labels, num_samples=20):
    """
    Display actual samples of the dataset for better understanding.

    - For tabular datasets: Prints and displays a dataframe preview.
    - For image datasets: Displays sample images with labels.

    Parameters:
        train_data (numpy.ndarray): Training feature set
        test_data (numpy.ndarray): Testing feature set
        train_labels (numpy.ndarray): Training labels
        test_labels (numpy.ndarray): Testing labels
        num_samples (int): Number of samples to display (default: 20)
    """

    # Print header for the function
    print("\n🎯 Visualize Dataset 🎯")

    # Ensure num_samples does not exceed dataset size
    num_samples = min(num_samples, len(train_data), len(test_data))

    if train_data.ndim == 2:
        # Tabular Dataset (e.g., Boston Housing)
        print("\n🔹 Train Data Sample:\n", pd.DataFrame(train_data[:num_samples]))
        print("\n🔹 Test Data Sample:\n", pd.DataFrame(test_data[:num_samples]))
        print("\n🔹 Train Labels Sample:\n", train_labels[:num_samples])
        print("\n🔹 Test Labels Sample:\n", test_labels[:num_samples])

    elif train_data.ndim == 3:
        # Image Dataset (e.g., MNIST)
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 5))
        axes = axes.flatten()

        for i in range(num_samples):
            axes[i].imshow(train_data[i], cmap="gray")
            axes[i].set_title(f"Label: {train_labels[i]}")
            axes[i].axis("off")

        plt.suptitle("Sample Images from Training Set\n")
        plt.show()

        print("\n🔹 Train Labels Sample:\n", train_labels[:num_samples])
        print("\n🔹 Test Labels Sample:\n", test_labels[:num_samples])

    else:
        print("⚠️ Unsupported data shape. Only 2D (tabular) and 3D (image) datasets are supported.")


# Function to visualize model training history
def visualize_history(history):
    """
    Plots the training and validation metrics of a Keras model.

    Parameters:
    model_history (History): The History object returned by the fit method of a Keras model.
    """

    # Print header for the function
    print("\n🎯 Visualize History 🎯")

    # Convert the history.history dictionary to a DataFrame
    history_df = pd.DataFrame(history.history)

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

# Print confirmation message
print("\n✅ visualize.py successfully executed")