# Import necessary libraries
import matplotlib.pyplot as plt

# Function to evaluate a regression model (actual vs. predicted)
def evaluate_model(model, train_data, train_labels, test_data, test_labels):
    """
    Visualize actual vs. predicted values for both training and test datasets.

    Parameters:
        model: Trained regression model (callable or with `predict()` method)
        train_data (numpy.ndarray): Training feature set
        test_data (numpy.ndarray): Testing feature set
        train_labels (numpy.ndarray): Training labels
        test_labels (numpy.ndarray): Testing labels
    """

    # Print header for the function
    print("\n🎯 Model Evaluation 🎯\n")

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

# Function to calculate the accuracy of a regression model
def calculate_model_accuracy(model, test_data, test_labels, threshold):
    """
    Calculate the accuracy of a regression model by comparing predictions with actual values.

    Parameters:
        model: Trained regression model (callable or with `predict()` method)
        test_data (numpy.ndarray): Testing feature set
        test_labels (numpy.ndarray): Testing labels
        error_threshold (float): Threshold for considering a prediction as an error

    Returns:
        accuracy (float): The accuracy of the model
        num_errors (int): The number of errors above the threshold
    """

    # Print header for the function
    print("\n🎯 Model Accuracy Calculation 🎯\n")

    # Predict values
    model_predictions = model.predict(test_data)

    # Initialize error counter
    num_errors = 0

    # Iterate over predictions and compare with actual values
    print(f"\n🔹 Model errors above {threshold} (threshold):\n")
    for index in range(len(model_predictions)):
        if abs(model_predictions[index] - (test_labels[index])) > threshold:
            print(f"Prediction: {model_predictions[index]}, Actual: {test_labels[index]}")
            num_errors += 1

    # Calculate accuracy
    accuracy = 1.0 - (num_errors / len(model_predictions))

    # Print summary
    print("\n🔹 Model Accuracy Summary:\n")
    print(f"Number of errors: {num_errors}")
    print(f"Accuracy: {accuracy:.2f}")

    # Return accuracy and number of errors
    return(accuracy)

# Print confirmation message
print("\n✅ evaluate.py successfully executed\n")