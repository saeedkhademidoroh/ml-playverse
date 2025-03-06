from data import load_dataset, analyze_dataset, preprocess_dataset
from models import build_model
from train import train_model
from evaluate import evaluate_model, calculate_model_accuracy
from visualize import visualize_model_history
from experiment import log_experiment
from config import BATCH_SIZE, EPOCHS, THRESHOLD

# Load & preprocess data
(train_data, train_labels), (test_data, test_labels) = load_dataset()
analyze_dataset(train_data, train_labels, test_data, test_labels)
train_data, train_labels, test_data, test_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

# Train and evaluate multiple models
for model_name in ["m1", "m2", "m3", "m4", "m5"]:
    model = build_model(model_name)
    history = train_model(model, train_data, train_labels, test_data, test_labels)
    evaluate_model(model, train_data, train_labels, test_data, test_labels)
    visualize_model_history(history)

    accuracy = calculate_model_accuracy(model, test_data, test_labels, THRESHOLD)
    log_experiment(model, history, accuracy, BATCH_SIZE, EPOCHS)
