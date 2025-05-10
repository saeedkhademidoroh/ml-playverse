# Project-specific imports
# from data import load_dataset, analyze_dataset, preprocess_dataset
# from evaluate import evaluate_model
from experiment import run_experiment
# from model import build_model
# from train import train_model


# Run experiment
# run_experiment((1, 6), runs=10, replace=True)

# Run Model 1 to 5, each 5 times
# run_experiment((1, 11), runs=5, replace=True)

# Run Model 3 one time
run_experiment(3)

# Run Models 3 to 5, each 5 times
# run_experiment((3, 5), runs=5)

# Run specific models 1, 3, and 5, each 2 times
# run_experiment([1, 3, 5], runs=2)


# Load dataset
# (train_data, train_labels), (test_data, test_labels) = load_dataset()

# Analyze dataset before preprocessing
# analyze_dataset(train_data, train_labels, test_data, test_labels)

# Preprocess dataset
# train_data, train_labels, test_data, test_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

# Analyze dataset after preprocessing
# analyze_dataset(train_data, train_labels, test_data, test_labels)

# Build model
# model, description = build_model(6)

# Train model
# model, history = train_model(train_data, train_labels, model, verbose=0)

# Evaluate model
# evaluation, predictions, shifted_evaluation, shifted_predictions = evaluate_model(test_data, test_labels, verbose=0)


# Print confirmation message
print("\n✅ main.py successfully executed")