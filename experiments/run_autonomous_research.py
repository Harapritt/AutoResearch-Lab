
"""Run bounded autonomous research experiments on AG News."""

import gc
import json
import shutil
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from agents.research_planner import ResearchPlanner
from configs.experiment_config import ExperimentConfig
from data.ag_news_loader import AGNewsLoader
from evaluation.metrics import ModelEvaluator
from utils.reproducibility import set_seed, get_device


def evaluate_model(model, dataloader, device):
    model.eval()

    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)
            predicted = outputs.logits.argmax(dim=1)

            predictions.extend(predicted.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return ModelEvaluator.evaluate(actual_labels, predictions)


def train_candidate(candidate, config, device):
    set_seed(config.random_seed)

    loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=candidate["max_length"]
    )

    dataloaders = loader.get_dataloaders(
        batch_size=candidate["batch_size"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=candidate["learning_rate"]
    )

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

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

        average_loss = total_loss / len(dataloaders["train"])

        print(
            f"{candidate['name']} | "
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"Loss: {average_loss:.4f}"
        )

    validation_metrics = evaluate_model(
        model,
        dataloaders["validation"],
        device
    )

    return model, loader.tokenizer, validation_metrics


def run_autonomous_research():
    config = ExperimentConfig()
    config.validate()

    device = get_device()
    print("Research device:", device)

    planner = ResearchPlanner()
    plan = planner.create_plan()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.checkpoint_dir) / "autonomous_best"

    history = []
    best_f1 = float("-inf")
    best_candidate = None

    print("\nResearch problem:", plan.research_problem)
    print("Hypothesis:", plan.hypothesis)
    print("\nStarting candidate experiments...")

    for index, candidate in enumerate(plan.experiment_candidates, start=1):
        print(
            f"\nExperiment {index}/{len(plan.experiment_candidates)}: "
            f"{candidate['name']}"
        )

        model = None

        try:
            model, tokenizer, metrics = train_candidate(
                candidate,
                config,
                device
            )

            result = {
                "experiment": candidate["name"],
                "parameters": candidate,
                "validation_metrics": metrics
            }

            history.append(result)

            print("Validation metrics:", metrics)

            current_f1 = metrics["f1_score"]

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_candidate = candidate.copy()

                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)

                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                print("New best candidate saved.")

        finally:
            if model is not None:
                del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_candidate is None:
        raise RuntimeError("No experiment completed successfully.")

    # Save experiment history before final evaluation.
    history_path = output_dir / "research_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # Reload the selected model and evaluate on the official test set.
    print("\nLoading best model for final test evaluation...")

    final_loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=best_candidate["max_length"]
    )

    final_dataloaders = final_loader.get_dataloaders(
        batch_size=best_candidate["batch_size"]
    )

    best_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir
    ).to(device)

    test_metrics = evaluate_model(
        best_model,
        final_dataloaders["test"],
        device
    )

    report = {
        "research_problem": plan.research_problem,
        "hypothesis": plan.hypothesis,
        "experiments_completed": len(history),
        "best_experiment": best_candidate,
        "best_validation_f1": best_f1,
        "final_test_metrics": test_metrics,
        "experiment_history": history
    }

    report_path = output_dir / "research_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nAutonomous research completed.")
    print("Best experiment:", best_candidate)
    print("Final test metrics:", test_metrics)
    print("History saved to:", history_path)
    print("Research report saved to:", report_path)
    print("Best model saved to:", checkpoint_dir)


if __name__ == "__main__":
    run_autonomous_research()
