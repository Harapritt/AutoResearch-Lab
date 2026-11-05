
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch


class AGNewsLoader:

    def __init__(self, model_name="bert-base-uncased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_data(self):
        self.dataset = load_dataset("fancyzhx/ag_news")
        return self.dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

    def get_dataloaders(self, batch_size=16):

        if self.dataset is None:
            self.load_data()

        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True
        )

        tokenized_dataset = tokenized_dataset.rename_column(
            "label", "labels"
        )

        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        train_loader = DataLoader(
            tokenized_dataset["train"],
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            tokenized_dataset["test"],
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, test_loader
