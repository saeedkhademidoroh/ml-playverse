# Standard imports
import sys
import os
import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd

# Project-specific imports
from config import CONFIG
from data import load_dataset, analyze_dataset, preprocess_dataset
from evaluate import evaluate_model
from model import build_model
from train import train_model


# Function to add experiment results to csv and xlsx files
def add_experiment_result(
    model,
    evaluation_result,
    description=None
):
    """
    Logs experiment parameters and results into CSV and XLSX files.

    Extracts key details such as:
    - Model name and architecture.
    - Training loss metrics.
    - Validation loss (if available).
    - Error count and accuracy.
    - Optimizer type.

    Parameters:
        model (tf.keras.Model): The trained model.
        history (tf.keras.callbacks.History): The training history.
        accuracy (float): The accuracy of model.
        error_count (int): The number of errors in model.
        description (str, optional): Additional description of experiment.

    Notes:
        - If CSV/XLSX files do not exist, they are created.
        - Ensures compatibility between new experiment data and existing records.
    """


    # Print header for function
    print("\n🎯 Add Experiment Result 🎯\n")

    # Extract model name
    name = model.name

    # Generate unique identifier using current date and time
    time = datetime.datetime.now().strftime("%H:%M:%S")

    # Extract model architecture details
    layers_count = len(model.layers)

    # Extract optimizer details
    optimizer = type(model.optimizer).__name__

    # Extract training metrics
    min_val_loss = evaluation_result["min_val_loss"]
    max_val_acc = evaluation_result["max_val_acc"]
    final_test_loss = evaluation_result["final_test_loss"]
    final_test_accuracy = evaluation_result["final_test_accuracy"]

    # Create dictionary of extracted data
    row_data = {
        "#": name,
        "Time": time,
        "Layers-Count": layers_count,
        "Optimizer": optimizer,
        "Min-Val-Loss": min_val_loss,
        "Max-Val-Acc": max_val_acc,
        "Fin-Test-Loss": final_test_loss,
        "Fin-Test-Acc": final_test_accuracy,
    }

    # Print values being logged
    print("🔹 Experiment Results:\n")
    for key, value in row_data.items():
        print(f"  {key}: {value}")

    # Get directory of current script
    CURRENT_DIR = Path(__file__).parent

    # Construct path to experiment results CSV file
    CSV_PATH = CURRENT_DIR / "experiment_results.csv"

    # Construct path to experiment results XLSX file
    EXCEL_PATH = CURRENT_DIR / "experiment_results.xlsx"

    # Load existing CSV or create new DataFrame
    try:
        experiment_results = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        experiment_results = pd.DataFrame(columns=row_data.keys())

    # Ensure new row and experiment_results have matching columns
    new_row = pd.DataFrame([row_data])
    for col in new_row.columns:
        if col not in experiment_results.columns:
            experiment_results[col] = pd.NA

    # Append new row to DataFrame
    experiment_results = pd.concat([new_row, experiment_results.dropna(axis=1, how="all")], ignore_index=True)

    # Write updated DataFrame to CSV and XLSX files
    with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter") as writer:
        experiment_results.to_excel(writer, index=False, sheet_name="Results")

        # Get xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets["Results"]

        # Set column widths based on max length of data in each column
        for col_idx, col in enumerate(experiment_results.columns):
            max_length = max(experiment_results[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(col_idx, col_idx, max_length)

        # Create cell format for centering text horizontally and vertically
        cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # Create bold cell format for header
        header_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True})

        # Write header with bold formatting
        for col_idx in range(len(experiment_results.columns)):
            worksheet.write(0, col_idx, experiment_results.columns[col_idx], header_format)

        # Write data rows with formatting (starting from row 1)
        for row_idx in range(len(experiment_results)):
            for col_idx in range(len(experiment_results.columns)):
                value = experiment_results.iloc[row_idx, col_idx]

                # Convert NaN/Inf to safe value
                if pd.isna(value): # Check for NaN
                    value = "N/A"
                elif value == np.inf: # Check for positive infinity
                    value = "Infinity"
                elif value == -np.inf: # Check for negative infinity
                    value = "-Infinity"

                worksheet.write(row_idx + 1, col_idx, value, cell_format)


    # Ensure directory exists and save file
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    experiment_results.to_csv(CSV_PATH, index=False)

# Function to run experiments for specified models
def run_experiment(model_numbers, runs=1, replace=True):
    """
    Trains and evaluates models based on specified configurations.

    Handles multiple model selections:
    - A single integer runs one specific model.
    - A tuple (start, end) runs all models in that range.
    - A list of integers runs specified models.

    Parameters:
        model_numbers (int | tuple[int, int] | list[int]):
            - Specifies which models to run.
        runs (int): Number of times to train each model (default: 1).
        replace (bool): If True, deletes existing output files before running.

    Notes:
        - Logs experiment progress to file.
        - Redirects standard output and errors to log file during execution.
        - Automatically restores normal console output after running.
    """


     # Print header for function
    print("\n🎯 Run Experiment 🎯\n")

    # Define paths
    current_dir = Path(__file__).parent
    csv_path = current_dir / "experiment_results.csv"
    xlsx_path = current_dir / "experiment_results.xlsx"
    log_path = current_dir / "experiment_log.txt"

    # Delete files if flag is set
    for file_path in [csv_path, xlsx_path, log_path]:
        if replace and file_path.exists():
            os.remove(file_path)
            print(f"Replacing {file_path.name} ...")
        else:
            print(f"Updating {file_path.name} ...")
    print("")

    # Open log file for writing
    with open(log_path, "a") as f:
        # Redirect stdout and stderr to file
        sys.stdout = f
        sys.stderr = f

        # Handle different model selection inputs
        if isinstance(model_numbers, int):
            model_numbers = [model_numbers] # Convert single model to list
        elif isinstance(model_numbers, tuple) and len(model_numbers) == 2:
            model_numbers = list(range(model_numbers[0], model_numbers[1] + 1)) # Create range

        # Load dataset
        (train_data, train_labels), (test_data, test_labels) = load_dataset()

        # Analyze dataset before preprocessing
        analyze_dataset(train_data, train_labels, test_data, test_labels)

        # Preprocess dataset
        train_data, train_labels, test_data, test_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

        # Analyze dataset after preprocessing
        analyze_dataset(train_data, train_labels, test_data, test_labels)

        for model_number in model_numbers:
            for run in range(1, runs + 1):
                print(f"\n🚀 Launching m{model_number} ({run}/{runs}) ...")

                # Build model
                model, description = build_model(model_number)

                # Train model
                model, history = train_model(train_data, train_labels, model)

                # Evaluate model
                evaluation_result = evaluate_model(model, history, test_data, test_labels)

                # Add experiment result
                add_experiment_result(model, evaluation_result, description)

        # Restore stdout and stderr to terminal
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


# Print confirmation message
print("\n✅ experiment.py successfully executed")