
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelLoader:

    def __init__(self, model_name, num_labels=4):
        self.model_name = model_name
        self.num_labels = num_labels

        self.tokenizer = None
        self.model = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        return self.tokenizer

    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        self.model.to(self.device)

        return self.model

    def load(self):
        self.load_tokenizer()
        self.load_model()

        return self.model, self.tokenizer


if __name__ == "__main__":
    loader = ModelLoader(
        model_name="distilbert-base-uncased",
        num_labels=4
    )

    model, tokenizer = loader.load()

    print("Model loaded successfully.")
    print("Device:", loader.device)
    print("Model:", loader.model_name)
