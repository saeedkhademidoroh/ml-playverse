# Standard library imports
import os
from pathlib import Path

# Third-party imports
from keras.api.callbacks import ModelCheckpoint

# Project-specific imports
from config import CONFIG


def train_model(train_data, train_labels, model, verbose=0):
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

    # Get directory of current script
    CURRENT_DIR = Path(__file__).parent

    # Construct path to file
    model_path = CURRENT_DIR / "trained_model.h5"

    # Train model and store training history
    print("\n🎯 Train Model 🎯")

    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=CONFIG.SAVE_BEST_ONLY)

    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE,
        validation_split=CONFIG.VALIDATION_SPLIT,
        callbacks=[model_checkpoint],
        verbose=verbose,
    )

    return model, history


# Print confirmation message
print("\n✅ train.py successfully executed")