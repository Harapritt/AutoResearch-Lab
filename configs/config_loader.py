import yaml
from pathlib import Path
from configs.experiment_config import ExperimentConfig


class ConfigLoader:

    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with path.open("r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a YAML mapping.")

        config = ExperimentConfig(**config_data)
        config.validate()

        return config