# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import load_model

# Project-specific imports
from config import CONFIG


# Function to extract min/max loss & accuracy from history
def extract_history_metrics(history):
    """
    Extracts min/max loss and accuracy from training history.

    Parameters:
        history (tf.keras.callbacks.History or dict): The training history.

    Returns:
        dict: Contains min/max loss & accuracy with their corresponding epochs.
    """


    # Print header for function
    print("\n🎯 Extract History 🎯")


    # Ensure history is a dictionary
    history = history.history if hasattr(history, "history") else history

    # Extract min/max values and corresponding epochs
    metrics = {
        "min_train_loss": min(history["loss"]),
        "min_train_loss_epoch": history["loss"].index(min(history["loss"])) + 1,
        "max_train_acc": max(history["accuracy"]),
        "max_train_acc_epoch": history["accuracy"].index(max(history["accuracy"])) + 1,
    }

    # Check for validation data (optional)
    if "val_loss" in history and "val_accuracy" in history:
        metrics.update({
            "min_val_loss": min(history["val_loss"]),
            "min_val_loss_epoch": history["val_loss"].index(min(history["val_loss"])) + 1,
            "max_val_acc": max(history["val_accuracy"]),
            "max_val_acc_epoch": history["val_accuracy"].index(max(history["val_accuracy"])) + 1,
        })
    else:
        metrics["min_val_loss"], metrics["max_val_acc"] = None, None

    return metrics


# Function to evaluate model
def evaluate_model(test_data, test_labels, verbose=0):
    """
    Evaluates the model on test data, extracts training history, and displays key metrics.

    Parameters:
        model (tf.keras.Model): A trained model.
        history (tf.keras.callbacks.History or dict): Training history.
        test_data (numpy.ndarray): Feature set for test data.
        test_labels (numpy.ndarray): True labels for test data.
        verbose (int): Verbosity level for model evaluation (default: 1).

    Returns:
        dict: Contains min/max loss, accuracy, test loss, test accuracy, and predictions.
    """

    print("\n🎯 Evaluate Model 🎯")

    # Load trained model
    model = load_model(CONFIG.MODEL_PATH)

    # Evaluate trained model
    evaluation = model.evaluate(test_data, test_labels, verbose=verbose)

    # Print evaluation results
    print("\n🔹 Evaluation Result:\n")
    print(evaluation)

    # Use trained model for predictions
    predictions = model.predict(test_data)

    # Print predictions
    print("\n🔹 Predictions:\n")
    for i in range(5):
        print(f"Sample {i + 1}:")
        print(f"Predicted: {np.argmax(predictions[i])}, True: {test_labels[i]}")

    return {
        "evaluation": evaluation,
        "predictions": predictions,
    }


# Print confirmation message
print("\n✅ evaluate.py successfully executed")