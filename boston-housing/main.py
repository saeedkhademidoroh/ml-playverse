# Project-specific imports
from experiment import run_experiment

# Run Model 1 to 5, each 5 times
run_experiment((1, 11), runs=5, replace=True)

# Run Model 3 one time
# run_experiment(3)

# Run Models 3 to 5, each 5 times
# run_experiment((3, 5), runs=5)

# Run specific models 1, 3, and 5, each 2 times
# run_experiment([1, 3, 5], runs=2)

# Print confirmation message
print("\n✅ main.py successfully executed")