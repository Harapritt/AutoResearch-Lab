
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class AGNewsLoader:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        max_length=256,
        validation_size=0.1,
        random_seed=42
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.validation_size = validation_size
        self.random_seed = random_seed

    def load_data(self):
        return load_dataset("fancyzhx/ag_news")

    def tokenize_data(self, dataset):

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        dataset = dataset.map(tokenize, batched=True)

        dataset = dataset.rename_column("label", "labels")

        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        return dataset

    def get_dataloaders(self, batch_size=16):

        dataset = self.load_data()

        # Split the original training data into train and validation.
        split = dataset["train"].train_test_split(
            test_size=self.validation_size,
            seed=self.random_seed,
            stratify_by_column="label"
        )

        # Keep the official test set separate.
        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            "test": dataset["test"]
        })

        dataset = self.tokenize_data(dataset)

        return {
            "train": DataLoader(
                dataset["train"],
                batch_size=batch_size,
                shuffle=True
            ),
            "validation": DataLoader(
                dataset["validation"],
                batch_size=batch_size,
                shuffle=False
            ),
            "test": DataLoader(
                dataset["test"],
                batch_size=batch_size,
                shuffle=False
            )
        }
