# Import necessary libraries
import os
import datetime
import numpy as np
import pandas as pd

# Function to add experiment results to a csv file
def add_experiment_result(
    train_data,
    train_labels,
    test_data,
    test_labels,
    model,
    batch_size,
    epochs,
    model_history,
    threshold,
    accuracy,
    description=None
):
    """
    Extracts experiment parameters and results from the model and history,
    then logs them into a CSV file.
    """

    # Print header for the function
    print("\n🎯 Experiment Results Logging 🎯\n")

    # Extract model name
    model_name = model.name

    # Generate a unique identifier using current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract training parameters
    learning_rate = getattr(model.optimizer, "learning_rate", None)
    if hasattr(learning_rate, "numpy"):
        learning_rate = learning_rate.numpy()  # Convert Tensor to float

    # Extract optimizer details
    optimizer = type(model.optimizer).__name__

    # Extract model architecture details
    dense_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    if dense_layers:
        activation_function = dense_layers[0].activation.__name__
        num_layers = len(dense_layers)
        num_units = dense_layers[0].units
    else:
        activation_function = None
        num_layers = len(model.layers)
        num_units = None

    # Extract evaluation metrics
    final_loss = model_history.history["loss"][-1]
    min_loss = min(model_history.history["loss"])
    max_loss = max(model_history.history["loss"])
    final_val_loss = model_history.history.get("val_loss", [None])[-1]

    # Create a dictionary of the extracted data
    row_data = {
        "Name": model_name,
        "Timestamp": timestamp,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Learning Rate": learning_rate,
        "Optimizer": optimizer,
        "Activation Function": activation_function,
        "Number of Layers": num_layers,
        "Number of Units": num_units,
        "Loss": final_loss,
        "Minimum Loss": min_loss,
        "Maximum Loss": max_loss,
        "Validation Loss": final_val_loss,
        "Error Threshold": threshold,
        "Accuracy": accuracy,
        "Description": description
    }

    # Print values being logged
    print("\n🔹 Experiment Results:\n")
    for key, value in row_data.items():
        print(f"  {key}: {value}")

    # Define CSV file path
    csv_path = os.path.expanduser("/home/saeed/projects/ml/src/mr-engineer-playverse/boston-housing/experiment_results.csv")

    # Load existing CSV or create new DataFrame
    try:
        experiment_results = pd.read_csv(csv_path)
    except FileNotFoundError:
        experiment_results = pd.DataFrame(columns=row_data.keys())

    # Ensure new row and experiment_results have matching columns
    new_row = pd.DataFrame([row_data])
    for col in new_row.columns:
        if col not in experiment_results.columns:
            experiment_results[col] = pd.NA

    # Append new row to the DataFrame
    experiment_results = pd.concat([new_row, experiment_results.dropna(axis=1, how="all")], ignore_index=True)

    # Save the updated DataFrame to CSV and Excel
    excel_path = csv_path.replace(".csv", ".xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
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
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    experiment_results.to_csv(csv_path, index=False)

# Print confirmation message
print("\n✅ experiment.py successfully executed\n")