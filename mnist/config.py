# Standard imports
import json  # JSON file handling
from pathlib import Path  # Path operations
from dataclasses import dataclass  # Immutable data structures


@dataclass(frozen=True)  # This makes the dataclass immutable
class Config:
    BATCH_SIZE: int
    EPOCHS: int
    THRESHOLD: float
    PATIENCE: int

    @staticmethod
    def load_from_json() -> "Config":
        """Loads configuration from a JSON file and ensures all required fields are present."""

        # Get the directory of the current script
        CURRENT_DIR = Path(__file__).parent

        # Construct the path to config.json
        CONFIG_PATH = CURRENT_DIR / "config.json"

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r") as file:
            config_data = json.load(file)

        # Required keys
        required_keys = ["BATCH_SIZE", "EPOCHS", "THRESHOLD", "PATIENCE"]
        missing_keys = [key for key in required_keys if key not in config_data]

        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        return Config(
            BATCH_SIZE=config_data["BATCH_SIZE"],
            EPOCHS=config_data["EPOCHS"],
            THRESHOLD=config_data["THRESHOLD"],
            PATIENCE=config_data["PATIENCE"],
        )


# Load from JSON (STRICT: Must exist & contain all required keys)
CONFIG = Config.load_from_json()

# Print confirmation message
print("\n✅ config.py successfully executed\n")
