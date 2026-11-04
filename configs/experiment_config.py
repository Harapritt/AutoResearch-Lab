
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    experiment_name: str = "transformer_baseline"

    dataset_name: str = "ag_news"
    model_name: str = "distilbert-base-uncased"

    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 256

    random_seed: int = 42

    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    def validate(self):
        if self.num_epochs <= 0:
            raise ValueError("Epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if self.max_length <= 0:
            raise ValueError("Maximum sequence length must be positive.")

        return True


if __name__ == "__main__":
    config = ExperimentConfig()

    config.validate()

    print("Experiment Configuration")
    print(config)
