
from datasets import load_dataset
from transformers import AutoTokenizer


class AGNewsLoader:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        max_length=256
    ):
        self.model_name = model_name
        self.max_length = max_length

        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

    def load_data(self):
        self.dataset = load_dataset("ag_news")
        return self.dataset

    def preprocess(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

    def tokenize_data(self):
        if self.dataset is None:
            self.load_data()

        tokenized_dataset = self.dataset.map(
            self.preprocess,
            batched=True
        )

        tokenized_dataset = tokenized_dataset.rename_column(
            "label", "labels"
        )

        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        return tokenized_dataset


if __name__ == "__main__":
    loader = AGNewsLoader()

    dataset = loader.tokenize_data()

    print("Dataset loaded successfully.")
    print(dataset)
    print("Training samples:", len(dataset["train"]))
    print("Test samples:", len(dataset["test"]))
