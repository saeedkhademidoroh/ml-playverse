import pandas as pd
import datetime
from config import EXPERIMENT_RESULTS_CSV

def log_experiment(model, history, accuracy, batch_size, epochs, description=None):
    """Logs experiment results to CSV."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_loss = history.history["loss"][-1]

    row = {
        "Model": model.name,
        "Timestamp": timestamp,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Final Loss": final_loss,
        "Accuracy": accuracy,
        "Description": description
    }

    try:
        df = pd.read_csv(EXPERIMENT_RESULTS_CSV)
    except FileNotFoundError:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([pd.DataFrame([row]), df])
    df.to_csv(EXPERIMENT_RESULTS_CSV, index=False)
