# Third-party imports
import matplotlib.pyplot as plt # Plotting and data visualization

# Project-specific imports
from config import CONFIG # Load configuration settings


# Function to evaluate model (actual vs. predicted)
def evaluate_model(model, train_data, train_labels, test_data, test_labels):
    """
    Visualizes actual vs. predicted values for both training and test datasets.

    This function generates side-by-side plots comparing true and predicted
    values for training and test datasets, which helps in evaluating how
    well model is performing.

    Parameters:
        model (tf.keras.Model): A trained model that has `predict()` method.
        train_data (numpy.ndarray): Feature set for training data.
        train_labels (numpy.ndarray): True labels for training data.
        test_data (numpy.ndarray): Feature set for test data.
        test_labels (numpy.ndarray): True labels for test data.

    Returns:
        None: This function does not return any values; it displays plots directly.
    """


    # Print header for function
    print("\n🎯 Evaluate Model 🎯\n")

    # Predict values
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)

    # Number of samples to visualize
    num_samples = min(30, len(train_labels), len(test_labels))

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot train data
    axes[0].plot(train_labels[:num_samples], "r-", label="True", alpha=0.7)
    axes[0].plot(train_preds[:num_samples], "b-", label="Predicted", alpha=0.7)
    axes[0].set_title("Train Data: Actual vs. Predicted")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot test data
    axes[1].plot(test_labels[:num_samples], "r-", label="True", alpha=0.7)
    axes[1].plot(test_preds[:num_samples], "b-", label="Predicted", alpha=0.7)
    axes[1].set_title("Test Data: Actual vs. Predicted")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Display plots
    plt.tight_layout()
    plt.show()


# Function to calculate accuracy of model
def calculate_model_accuracy(model, test_data, test_labels):
    """
    Calculates accuracy of model by comparing predictions with actual values.

    This function computes accuracy of model based on number of
    errors (absolute difference between predicted and true values) exceeding
    given threshold. The accuracy is then calculated as proportion of
    correct predictions.

    Parameters:
        model (tf.keras.Model): A trained model that has `predict()` method.
        test_data (numpy.ndarray): Feature set for test data.
        test_labels (numpy.ndarray): True labels for test data.

    Returns:
        tuple:
            - error_count (int): The number of predictions with errors exceeding threshold.
            - accuracy (float): The proportion of correct predictions, calculated as (1 - error_count / total predictions).
    """


    # Print header for function
    print("\n🎯 Calculate Model Accuracy 🎯")

    # Predict values
    model_predictions = model.predict(test_data)

    # Initialize error counter
    error_count = 0

    # Iterate over predictions and compare with actual values
    print(f"\n🔹 Model errors above {CONFIG.THRESHOLD} (threshold):\n")
    for index in range(len(model_predictions)):
        if abs(model_predictions[index] - (test_labels[index])) > CONFIG.THRESHOLD:
            print(f"Prediction: {model_predictions[index]}, Actual: {test_labels[index]}")
            error_count += 1

    # Calculate accuracy
    accuracy = 1.0 - (error_count / len(model_predictions))

    # Print summary
    print("\n🔹 Model Accuracy Summary:\n")
    print(f"Number of errors: {error_count}")
    print(f"Accuracy: {accuracy:.2f}")

    # Return accuracy and number of errors
    return(error_count, accuracy)


# Print confirmation message
print("\n✅ evaluate.py successfully executed")