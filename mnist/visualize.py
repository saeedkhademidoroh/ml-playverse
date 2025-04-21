# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 5))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(train_data[i], cmap="gray")
        axes[i].set_title(f"Label: {train_labels[i]}")
        axes[i].axis("off")

    plt.suptitle("Sample Images from Training Set\n")
    plt.show()

    print("\n🔹 Train Labels Sample:\n\n", train_labels[:num_samples])
    print("\n🔹 Test Labels Sample:\n\n", test_labels[:num_samples])


# Function to visualize model training history
def visualize_history(history):
    """
    Plots training and validation metrics of a Keras model.

    Displays:
    - Training loss vs. Validation loss across epochs.
    - Training accuracy vs. Validation accuracy across epochs (if available).
    - Annotates min loss and max accuracy in the plot.

    Parameters:
        history (tf.keras.callbacks.History): The History object from model.fit().

    Notes:
        - Only plots accuracy if "accuracy" and "val_accuracy" exist in history.
        - Loss and accuracy are displayed in separate subplots for clarity.
        - Prints min/max metrics to the terminal.
    """

    # Print header for function
    print("\n🎯 Visualize Training History 🎯")

    # Convert history.history dictionary to DataFrame
    history_df = pd.DataFrame(history.history)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----------- Loss Plot -----------
    loss_col = 'loss'
    val_loss_col = 'val_loss'

    # Find min loss values
    min_loss_idx = history_df[loss_col].idxmin()
    min_val_loss_idx = history_df[val_loss_col].idxmin()

    min_loss = history_df[loss_col][min_loss_idx]
    min_val_loss = history_df[val_loss_col][min_val_loss_idx]

    # Plot loss
    history_df[[loss_col, val_loss_col]].rename(columns={
        loss_col: 'Training Loss',
        val_loss_col: 'Validation Loss'
    }).plot(ax=axes[0])
    axes[0].set_title("Loss Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Add markers for min loss
    axes[0].scatter(min_loss_idx, min_loss, color="red", label=f"Min Loss: {min_loss:.4f}")
    axes[0].scatter(min_val_loss_idx, min_val_loss, color="blue", label=f"Min Val Loss: {min_val_loss:.4f}")
    axes[0].legend()

    # ----------- Accuracy Plot -----------
    acc_col = "accuracy"
    val_acc_col = "val_accuracy"

    # Find max accuracy values
    max_acc_idx = history_df[acc_col].idxmax()
    max_val_acc_idx = history_df[val_acc_col].idxmax()

    max_acc = history_df[acc_col][max_acc_idx]
    max_val_acc = history_df[val_acc_col][max_val_acc_idx]

    # Plot accuracy
    history_df[[acc_col, val_acc_col]].rename(columns={
        acc_col: 'Training Accuracy',
        val_acc_col: 'Validation Accuracy'
    }).plot(ax=axes[1])
    axes[1].set_title("Accuracy Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    # Add markers for max accuracy
    axes[1].scatter(max_acc_idx, max_acc, color="red", label=f"Max Acc: {max_acc:.4f}")
    axes[1].scatter(max_val_acc_idx, max_val_acc, color="blue", label=f"Max Val Acc: {max_val_acc:.4f}")
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def visualize_predictions(test_data, test_labels, predictions, num_samples=20):
    """
    Plots test images with their true and predicted labels.

    Parameters:
        test_data (numpy.ndarray): Test images (MNIST: 28x28 grayscale).
        test_labels (numpy.ndarray): True labels.
        predictions (numpy.ndarray): Model's predicted probabilities.
        num_samples (int): Number of samples to display.
    """


    # Print header for function
    print("\n🎯 Visualize Predictions 🎯")

    # Select random indices for visualization
    sample_indices = np.random.choice(len(test_labels), num_samples, replace=False)
    sample_images = test_data[sample_indices]
    sample_labels = test_labels[sample_indices]
    sample_preds = np.argmax(predictions[sample_indices], axis=1)

    # Determine grid size (1 row if ≤5, else multiple rows)
    cols = min(num_samples, 5)  # Max 5 per row
    rows = (num_samples + cols - 1) // cols  # Calculate needed rows

    # Create subplots with dynamic grid size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).flatten()  # Flatten in case of 1 row

    # Plot each sample
    for i in range(num_samples):
        axes[i].imshow(sample_images[i].reshape(28, 28), cmap="gray")
        axes[i].set_title(f"Pred: {sample_preds[i]} | True: {sample_labels[i]}", fontsize=10)
        axes[i].axis("off")

    # Hide any unused subplots (for non-exact grids)
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# Print confirmation message
print("\n✅ visualize.py successfully executed")