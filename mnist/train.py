# Project-specific imports
from config import CONFIG  # Configurations for the project
from keras.api.callbacks import EarlyStopping  # Early stopping callback for model training


def train_model(train_data, train_labels, test_data, test_labels, model):
    """
    Trains the given model and returns the trained model along with its training history.

    Parameters:
    - train_data (numpy.ndarray): Training features.
    - train_labels (numpy.ndarray): Training labels.
    - test_data (numpy.ndarray): Testing features.
    - test_labels (numpy.ndarray): Testing labels.
    - model (tf.keras.Model): The model to train.

    Returns:
    - model (tf.keras.Model): The trained model.
    - history (tf.keras.callbacks.History): Training history.
    """

    # Train the model and store the training history
    print("\n🎯 Train Model 🎯\n")

    # Early stopping callback
    early_stop = EarlyStopping(monitor="val_loss", patience=CONFIG.PATIENCE, restore_best_weights=True)

    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS,  # Taken from config.py
        batch_size=CONFIG.BATCH_SIZE,  # Taken from config.py
        validation_data=(test_data, test_labels),
        callbacks=[early_stop]
    )

    return model, history

# Print confirmation message
print("\n✅ train.py successfully executed")