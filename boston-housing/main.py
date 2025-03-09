# Project specific imports
from data import load_dataset, analyze_dataset, preprocess_dataset  # Dataset loading, analysis, and preprocessing
from model import build_model  # Model building function
from train import train_model  # Function for model training
from evaluate import evaluate_model, calculate_model_accuracy  # Model evaluation and accuracy calculation
from visualize import visualize_history  # Visualization of model training history
from experiment import add_experiment_result  # Logging experiment results


# Load dataset and split into training and test sets
(train_data, train_labels), (test_data, test_labels) = load_dataset()

# Analyze dataset before preprocessing
analyze_dataset(train_data, train_labels, test_data, test_labels)

# Preprocess dataset
train_data, train_labels, test_data, test_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

# Analyze dataset after preprocessing
analyze_dataset(train_data, train_labels, test_data, test_labels)

# Hardcoded model selection
model = build_model(1)

# Train the model
model, history = train_model(train_data, train_labels, test_data, test_labels, model)

# Evaluate model performance
evaluate_model(model, train_data, train_labels, test_data, test_labels)

# Visualize model training history
visualize_history(history)

# Calculate model accuracy
error_count, accuracy = calculate_model_accuracy(
    model,
    test_data,
    test_labels,
)

# Log experiment results
add_experiment_result(
    model,
    history,
    accuracy,
    error_count
)

# Print confirmation message
print("\n✅ main.py successfully executed")

# Print the log message
print("\n🔹 Empty log message")
