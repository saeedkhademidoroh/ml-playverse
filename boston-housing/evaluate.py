import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, train_data, train_labels, test_data, test_labels):
    """Plots actual vs predicted values."""
    print("\n🎯 Evaluating Model 🎯\n")

    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)

    plt.figure(figsize=(12, 5))
    plt.plot(train_labels[:30], "r-", label="Actual")
    plt.plot(train_preds[:30], "b-", label="Predicted")
    plt.title("Train: Actual vs. Predicted")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(test_labels[:30], "r-", label="Actual")
    plt.plot(test_preds[:30], "b-", label="Predicted")
    plt.title("Test: Actual vs. Predicted")
    plt.legend()
    plt.show()

def calculate_model_accuracy(model, test_data, test_labels, threshold=5.0):
    """Computes accuracy based on a prediction threshold."""
    predictions = model.predict(test_data)
    num_errors = np.sum(np.abs(predictions - test_labels) > threshold)
    accuracy = 1 - (num_errors / len(predictions))

    print(f"Accuracy: {accuracy:.2f}")
    return accuracy
