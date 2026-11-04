import time
import torch
import mlflow

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from configs.experiment_config import ExperimentConfig
from data.ag_news_loader import AGNewsLoader
from utils.reproducibility import set_seed, get_device
from evaluation.metrics import ModelEvaluator


def run_experiment():

    config = ExperimentConfig()
    config.validate()

    set_seed(config.random_seed)
    device = get_device()

    print("Using device:", device)

    loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=config.max_length
    )

    dataloaders = loader.get_dataloaders(
        batch_size=config.batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    mlflow.set_experiment(config.experiment_name)

    start_time = time.time()

    with mlflow.start_run():

        mlflow.log_params({
            "model": config.model_name,
            "dataset": config.dataset_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.num_epochs,
            "max_length": config.max_length,
            "seed": config.random_seed
        })

        for epoch in range(config.num_epochs):

            model.train()
            total_loss = 0

            for batch in dataloaders["train"]:

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                optimizer.zero_grad()

                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            average_loss = (
                total_loss / len(dataloaders["train"])
            )

            print(
                f"Epoch {epoch + 1}: "
                f"Training Loss = {average_loss:.4f}"
            )

            mlflow.log_metric(
                "training_loss",
                average_loss,
                step=epoch + 1
            )

        model.eval()

        predictions = []
        actual_labels = []

        with torch.no_grad():

            for batch in dataloaders["test"]:

                labels = batch["labels"].to(device)

                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key != "labels"
                }

                outputs = model(**inputs)

                predictions.extend(
                    outputs.logits.argmax(dim=1).cpu().tolist()
                )

                actual_labels.extend(
                    labels.cpu().tolist()
                )

        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(
            actual_labels,
            predictions
        )

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        duration = time.time() - start_time

        mlflow.log_metric("duration_seconds", duration)

        save_path = config.checkpoint_dir

        model.save_pretrained(save_path)
        loader.tokenizer.save_pretrained(save_path)

        mlflow.log_artifacts(save_path, artifact_path="model")

        print("\nExperiment completed.")
        print("Evaluation metrics:", metrics)
        print("Model saved to:", save_path)


if __name__ == "__main__":
    run_experiment()