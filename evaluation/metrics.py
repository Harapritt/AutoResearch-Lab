
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    @staticmethod
    def evaluate(y_true, y_pred):

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        }

        return results

    @staticmethod
    def detailed_report(y_true, y_pred):
        return classification_report(
            y_true,
            y_pred,
            zero_division=0
        )

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    actual = [0, 1, 2, 1, 0, 2]
    predicted = [0, 1, 2, 0, 0, 2]

    evaluator = ModelEvaluator()

    results = evaluator.evaluate(actual, predicted)

    print("Model Evaluation Results")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(evaluator.detailed_report(actual, predicted))