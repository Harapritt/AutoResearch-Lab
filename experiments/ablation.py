import torch
import pandas as pd

from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from configs.experiment_config import ExperimentConfig
from data.ag_news_loader import AGNewsLoader
from utils.reproducibility import set_seed, get_device


def run_ablation(max_length):

    config = ExperimentConfig()
    config.max_length = max_length

    set_seed(config.random_seed)
    device = get_device()

    loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=max_length
    )

    dataloaders = loader.get_dataloaders(
        batch_size=config.batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    for epoch in range(config.num_epochs):

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

    return {
        "max_length": max_length,
        "accuracy": accuracy
    }


def main():

    results = []

    for length in [128, 256, 512]:

        print(f"Running experiment with max_length={length}")

        result = run_ablation(length)
        results.append(result)

        print(result)

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        "ablation_results.csv",
        index=False
    )

    print("\nAblation results:")
    print(dataframe)


if __name__ == "__main__":
    main()