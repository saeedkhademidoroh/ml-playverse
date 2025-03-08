# Import necessary libraries
import tensorflow as tf

# Function to train the model
def train_model(model, train_data, train_labels, test_data, test_labels, epochs=50, batch_size=32):
    """
    Trains the model and returns the training history.

    Parameters:
    - model (tf.keras.Model): The compiled model.
    - train_data (numpy.ndarray): Training features.
    - train_labels (numpy.ndarray): Training labels.
    - test_data (numpy.ndarray): Testing features.
    - test_labels (numpy.ndarray): Testing labels.
    - epochs (int): Number of training epochs (default: 50).
    - batch_size (int): Batch size for training (default: 32).

    Returns:
    - history (tf.keras.callbacks.History): Training history.
    """

    print("\n▶ Training the model...\n")

    # Train the model and store the training history
    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_labels)
        # callbacks=[early_stop]  # Uncomment if you have early stopping
    )  # type: ignore

    return history


# Print confirmation message
print("\n✅ train.py successfully executed\n")