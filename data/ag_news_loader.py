from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class AGNewsLoader:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        max_length=256
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.max_length = max_length

    def load_data(self):
        return load_dataset("ag_news")

    def tokenize_data(self, dataset):

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        dataset = dataset.map(
            tokenize,
            batched=True
        )

        dataset = dataset.rename_column(
            "label",
            "labels"
        )

        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "labels"
            ]
        )

        return dataset

    def get_dataloaders(self, batch_size=16):

        dataset = self.load_data()
        dataset = self.tokenize_data(dataset)

        return {
            "train": DataLoader(
                dataset["train"],
                batch_size=batch_size,
                shuffle=True
            ),
            "test": DataLoader(
                dataset["test"],
                batch_size=batch_size
            )
        }