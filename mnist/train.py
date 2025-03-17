# Project-specific imports
from config import CONFIG # Configurations for project
from keras.api.callbacks import EarlyStopping # Early stopping callback for model training


def train_model(train_data, train_labels, test_data, test_labels, model):
    """
    Trains given model using provided training data and labels.

    Training includes:
    - Early stopping to prevent overfitting.
    - Validation on test dataset.
    - Configurable training parameters (epochs, batch size) from CONFIG.

    Parameters:
        train_data (numpy.ndarray): Training features.
        train_labels (numpy.ndarray): Training labels.
        test_data (numpy.ndarray): Testing features.
        test_labels (numpy.ndarray): Testing labels.
        model (tf.keras.Model): The model to train.

    Returns:
        tuple:
            - model (tf.keras.Model): The trained model.
            - history (tf.keras.callbacks.History): Training history containing loss and accuracy metrics.
    """


    # Train model and store training history
    print("\n🎯 Train Model 🎯\n")

    # Early stopping callback
    early_stop = EarlyStopping(monitor="val_accuracy", patience=CONFIG.PATIENCE, restore_best_weights=True, mode="max")

    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS, # Taken from config.py
        batch_size=CONFIG.BATCH_SIZE, # Taken from config.py
        validation_data=(test_data, test_labels),
        callbacks=[early_stop]
    )

    return model, history


# Print confirmation message
print("\n✅ train.py successfully executed")