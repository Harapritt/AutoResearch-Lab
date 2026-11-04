
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from configs.experiment_config import ExperimentConfig
from datasets.ag_news_loader import AGNewsLoader
from evaluation.metrics import ModelEvaluator


def evaluate():

    config = ExperimentConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load test dataset
    dataset_loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=config.max_length
    )

    dataset = dataset_loader.tokenize_data()
    test_dataset = dataset["test"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Load trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint_dir
    )

    model.to(device)
    model.eval()

    evaluator = ModelEvaluator()

    actual_labels = []
    predicted_labels = []

    print("Starting evaluation...")

    with torch.no_grad():

        for batch in test_loader:

            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            actual_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    # Calculate metrics
    results = evaluator.evaluate(
        actual_labels,
        predicted_labels
    )

    print("\nEvaluation Results")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nDetailed Classification Report:")

    print(
        evaluator.detailed_report(
            actual_labels,
            predicted_labels
        )
    )


if __name__ == "__main__":
    evaluate()