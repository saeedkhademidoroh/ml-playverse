# Import standard libraries
import os

# Import project-specific modules
from experiment import run_pipeline
from log import clean_old_output


# Print module execution banner
print("\n✅  main.py is being executed")

# Clean old outputs if CLEAN_MODE is enabled
clean_old_output(False)

# Force CPU usage by disabling GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (0, "default"),
    (1, "default"),
    (2, "default"),
    (3, "default"),
    (4, "default"),
    (5, "default"),
    (6, "default"),
    (7, "default"),
    (8, "default")
]

# Run experiments through pipeline
run_pipeline(pipeline)
