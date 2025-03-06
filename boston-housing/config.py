import os

# Paths
DATA_PATH = "/home/saeed/projects/ml/src/mr-engineer-playverse/boston-housing/"
EXPERIMENT_RESULTS_CSV = os.path.join(DATA_PATH, "experiment_results.csv")

# Training settings
BATCH_SIZE = 8
EPOCHS = 200
THRESHOLD = 5.0

# Model hyperparameters
INPUT_SHAPE = (13,)
DEFAULT_OPTIMIZER = "adam"  # Can be "adam" or "sgd"
DEFAULT_LOSS = "mse"
