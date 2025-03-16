# Standard imports
import sys  # System-specific parameters and functions
import os  # File and directory operations
import datetime  # Date and time utilities
from pathlib import Path  # Path handling

# Third-party imports
import numpy as np  # Numerical computing
import pandas as pd  # Data manipulation

# Project-specific imports
from config import CONFIG  # Configuration settings
from data import load_dataset, preprocess_dataset  # Data loading and preprocessing
from evaluate import calculate_model_accuracy  # Model evaluation
from model import build_model  # Model building
from train import train_model  # Model training

# Function to add experiment results to csv and xlsx files
def add_experiment_result(
    model,
    history,
    accuracy,
    error_count,
    description=None
):
    """
    Extracts experiment parameters and results from the model and history,
    then logs them into csv and xlsx files

    Parameters:
    - model (tf.keras.Model): The trained model.
    - history (tf.keras.callbacks.History): The training history.
    - error_count (int): The number of errors in the model.
    - accuracy (float): The accuracy of the model.
    - description (str, optional): A description of the experiment (default: None).
    """


    # Print header for the function
    print("\n🎯 Add Experiment Result 🎯\n")

    # Extract model name
    name = model.name

    # Generate a unique identifier using current date and time
    time = datetime.datetime.now().strftime("%H:%M:%S")

    # Extract model architecture details
    layers_count = len(model.layers)

    # Extract optimizer details
    optimizer = type(model.optimizer).__name__

    # Extract evaluation metrics
    min_loss = min(history.history["loss"])
    final_loss = history.history["loss"][-1]
    val_loss = history.history.get("val_loss", [None])[-1]

    # Create a dictionary of the extracted data
    row_data = {
        "#": name,
        "Time": time,
        "L-Count": layers_count,
        "O-Type": optimizer,
        "M-Loss": int(min_loss),
        "F-Loss": int(final_loss),
        "V-Loss": int(val_loss),
        "E-Count": error_count,
        "Accuracy": int(accuracy * 100),
    }

    # Print values being logged
    print("🔹 Experiment Results:\n")
    for key, value in row_data.items():
        print(f"  {key}: {value}")

    # Get the directory of the current script
    CURRENT_DIR = Path(__file__).parent

    # Construct the path to the experiment results CSV file
    CSV_PATH = CURRENT_DIR / "experiment_results.csv"

    # Construct the path to the experiment results XLSX file
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

    # Append new row to the DataFrame
    experiment_results = pd.concat([new_row, experiment_results.dropna(axis=1, how="all")], ignore_index=True)

    # Write the updated DataFrame to the CSV and XLSX files
    with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter") as writer:
        experiment_results.to_excel(writer, index=False, sheet_name="Results")

        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets["Results"]

        # Set column widths based on the max length of the data in each column
        for col_idx, col in enumerate(experiment_results.columns):
            max_length = max(experiment_results[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(col_idx, col_idx, max_length)

        # Create a cell format for centering text horizontally and vertically
        cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # Create a bold cell format for the header
        header_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True})

        # Write header with bold formatting
        for col_idx in range(len(experiment_results.columns)):
            worksheet.write(0, col_idx, experiment_results.columns[col_idx], header_format)

        # Write data rows with formatting (starting from row 1)
        for row_idx in range(len(experiment_results)):
            for col_idx in range(len(experiment_results.columns)):
                value = experiment_results.iloc[row_idx, col_idx]

                # Convert NaN/Inf to a safe value
                if pd.isna(value):  # Check for NaN
                    value = "N/A"
                elif value == np.inf:  # Check for positive infinity
                    value = "Infinity"
                elif value == -np.inf:  # Check for negative infinity
                    value = "-Infinity"

                worksheet.write(row_idx + 1, col_idx, value, cell_format)


    # Ensure directory exists and save the file
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    experiment_results.to_csv(CSV_PATH, index=False)

# Function to run experiments for specified models
def run_experiment(model_numbers, runs=1, replace=False):
    """
    Runs training and evaluation for specified models.

    Parameters:
    - model_numbers (int | tuple[int, int] | list[int]):
        - If an integer is provided, runs the experiment for that model only.
        - If a tuple (start, end) is given, runs experiments for models in that range.
        - If a list is given, runs experiments for the specified models.
    - runs (int): Number of times to train each model (default: 1).
    - replace (bool): If True, deletes output files. If False, updates them. Default is False.
    """

     # Print header for the function
    print("\n🎯 Run Experiment 🎯\n")

    # Define paths
    current_dir = Path(__file__).parent
    csv_path = current_dir / "experiment_results.csv"
    xlsx_path = current_dir / "experiment_results.xlsx"
    log_path = current_dir / "experiment_log.txt"

    # Delete files if the flag is set
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
            model_numbers = [model_numbers]  # Convert single model to a list
        elif isinstance(model_numbers, tuple) and len(model_numbers) == 2:
            model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))  # Create range

        for model_number in model_numbers:
            for run in range(1, runs + 1):
                print(f"\n🚀 Launching m{model_number} ({run}/{runs}) ...")

                # Load and preprocess dataset
                (train_data, train_labels), (test_data, test_labels) = load_dataset()
                train_data, train_labels, test_data, test_labels = preprocess_dataset(train_data, train_labels, test_data, test_labels)

                # Build and train model
                model, description = build_model(model_number)
                model, history = train_model(train_data, train_labels, test_data, test_labels, model)

                # Evaluate and log results
                error_count, accuracy = calculate_model_accuracy(model, test_data, test_labels)
                add_experiment_result(model, history, accuracy, error_count, description)

        # Restore stdout and stderr to terminal
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


# Print confirmation message
print("\n✅ experiment.py successfully executed")