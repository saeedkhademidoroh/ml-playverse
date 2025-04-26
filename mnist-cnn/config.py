# Standard imports
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True) # This makes dataclass immutable
class Config:
    """
    A configuration class that holds hyperparameters for model training.

    This class is immutable and contains various hyperparameters used for model
    training and evaluation.

    Attributes:
        Any parameter that is specified in configuration JSON file.

    Methods:
        load_from_json() -> Config:
            Loads configuration from JSON file and returns Config instance.
            Ensures all required fields are present in file.
    """

    SAVE_BEST_ONLY: bool
    BATCH_SIZE: int
    EPOCHS: int
    VALIDATION_SPLIT: float
    MODEL_PATH: str

    @staticmethod
    def load_from_json() -> "Config":
        """
        Loads configuration from JSON file and ensures all required fields are present.

        This method reads JSON file named `config.json` in same directory
        as script, and validates that file contains necessary fields
        for configuration. If file is missing or any required fields are
        absent, appropriate exceptions are raised.

        Returns:
            Config: An immutable Config instance with values loaded from JSON file.

        Raises:
            FileNotFoundError: If `config.json` does not exist.
            ValueError: If any required keys are missing from file.
        """


        # Get directory of current script
        CURRENT_DIR = Path(__file__).parent

        # Construct path to config.json
        CONFIG_PATH = CURRENT_DIR / "config.json"

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r") as file:
            config_data = json.load(file)

        # Required keys
        required_keys = [
            "SAVE_BEST_ONLY",
            "BATCH_SIZE",
            "EPOCHS",
            "VALIDATION_SPLIT",
            "MODEL_PATH",
        ]
        missing_keys = [key for key in required_keys if key not in config_data]

        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        return Config(
            SAVE_BEST_ONLY=config_data["SAVE_BEST_ONLY"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            EPOCHS=config_data["EPOCHS"],
            VALIDATION_SPLIT=config_data["VALIDATION_SPLIT"],
            MODEL_PATH=config_data["MODEL_PATH"],
        )


# Load from JSON (STRICT: Must exist & contain all required keys)
CONFIG = Config.load_from_json()

# Print confirmation message
print("\n✅ config.py successfully executed")
