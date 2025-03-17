# Project-specific imports
from data import load_dataset # Function to load dataset
from visualize import visualize_dataset # Function to visualize dataset
from data import analyze_dataset # Function to analyze dataset
from data import preprocess_dataset # Function to preprocess dataset
from model import build_model
from train import train_model
from experiment import run_experiment # Function to run experiments


# Load dataset
(train_data, train_labels), (test_data, test_labels) = load_dataset()

# Visualize dataset
# visualize_dataset(train_data, train_labels, test_data, test_labels, num_samples=20)

# Analyze dataset
# analyze_dataset(train_data, train_labels, test_data, test_labels)

# Preprocess dataset
train_data, train_labels, test_data, test_labels, val_data, val_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

# Analyze dataset after preprocessing
# analyze_dataset(train_data, train_labels, test_data, test_labels)

# Build model
model, description = build_model(1)

# Train model
model, history = train_model(train_data, train_labels, val_data, val_labels, model)

# Run Model 1 to 5, each 5 times
# run_experiment((1, 11), runs=5, replace=True)

# Run Model 3 one time
# run_experiment(3)

# Run Models 3 to 5, each 5 times
# run_experiment((3, 5), runs=5)

# Run specific models 1, 3, and 5, each 2 times
# run_experiment([1, 3, 5], runs=2)

# Print confirmation message
print("\n✅ main.py successfully executed")