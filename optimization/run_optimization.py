import optuna
import torch

from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from configs.experiment_config import ExperimentConfig
from data.ag_news_loader import AGNewsLoader
from utils.reproducibility import set_seed, get_device


def objective(trial):

    config = ExperimentConfig()

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 5e-4, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [8, 16, 32]
    )

    set_seed(config.random_seed)
    device = get_device()

    loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=config.max_length
    )

    dataloaders = loader.get_dataloaders(
        batch_size=batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate
    )

    # Train for one epoch per trial.
    model.train()

    for batch in dataloaders["train"]:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

    # Evaluate on the test split.
    # For final research, use a separate validation split
    # for hyperparameter selection.
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloaders["test"]:

            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy


def run_optimization():

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=10
    )

    print("Best parameters:", study.best_params)
    print("Best accuracy:", study.best_value)

    study.trials_dataframe().to_csv(
        "optimization_results.csv",
        index=False
    )


if __name__ == "__main__":
    run_optimization()