# Third-party imports
import pandas as pd # Data manipulation with pandas
import matplotlib.pyplot as plt # Plotting library

# Function to visualize dataset
def visualize_dataset(train_data, train_labels, test_data, test_labels, num_samples=20):
    """
    Displays actual samples of dataset for better understanding.

    For tabular datasets:
    - Prints preview of dataset as pandas DataFrame.

    For image datasets:
    - Displays grid of sample images with corresponding labels.

    Parameters:
        train_data (numpy.ndarray): Training feature set.
        test_data (numpy.ndarray): Testing feature set.
        train_labels (numpy.ndarray): Training labels.
        test_labels (numpy.ndarray): Testing labels.
        num_samples (int): Number of samples to display (default: 20).

    Notes:
        - Only supports 2D (tabular) and 3D (image) datasets.
        - Ensures `num_samples` does not exceed dataset size.
    """


    # Print header for function
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
    Plots training and validation metrics of Keras model.

    Displays:
    - Training loss vs. Validation loss across epochs.

    Parameters:
        history (tf.keras.callbacks.History): The History object from model.fit().

    Notes:
        - The history dictionary keys are renamed for better readability.
        - A line plot is generated for visual comparison.
    """


    # Print header for function
    print("\n🎯 Visualize History 🎯")

    # Convert history.history dictionary to DataFrame
    history_df = pd.DataFrame(history.history)

    # Rename columns for better readability
    history_df.rename(columns={
        'loss': 'Training Loss',
        'val_loss': 'Validation Loss'
    }, inplace=True)

    # Plot DataFrame
    history_df.plot(figsize=(10, 6))
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)

    # Display plot
    plt.show()

# Print confirmation message
print("\n✅ visualize.py successfully executed")