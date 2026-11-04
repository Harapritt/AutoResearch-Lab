
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from configs.experiment_config import ExperimentConfig
from models.model_loader import ModelLoader
from datasets.ag_news_loader import AGNewsLoader


def train():

    config = ExperimentConfig()
    config.validate()

    torch.manual_seed(config.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    # Load dataset
    dataset_loader = AGNewsLoader(
        model_name=config.model_name,
        max_length=config.max_length
    )

    dataset = dataset_loader.tokenize_data()

    train_dataset = dataset["train"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Load model
    model_loader = ModelLoader(
        model_name=config.model_name,
        num_labels=4
    )

    model, tokenizer = model_loader.load()

    device = model_loader.device

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    print("Starting training...")

    for epoch in range(config.num_epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

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

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    # Save trained model
    model.save_pretrained(config.checkpoint_dir)
    tokenizer.save_pretrained(config.checkpoint_dir)

    print("Training completed.")
    print("Model saved to:", config.checkpoint_dir)


if __name__ == "__main__":
    train()