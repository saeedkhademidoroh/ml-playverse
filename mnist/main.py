# Project-specific imports
from data import load_dataset # Function to load the dataset
from visualize import visualize_dataset # Function to visualize the dataset
from data import analyze_dataset # Function to analyze the dataset
from data import preprocess_dataset # Function to preprocess the dataset
from experiment import run_experiment # Function to run experiments


# Load the dataset
(train_data, train_labels), (test_data, test_labels) = load_dataset()

# Visualize the dataset
# visualize_dataset(train_data, train_labels, test_data, test_labels, num_samples=20)

# Analyze the dataset
# analyze_dataset(train_data, train_labels, test_data, test_labels)

# Preprocess the dataset
train_data, train_labels, test_data, test_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

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