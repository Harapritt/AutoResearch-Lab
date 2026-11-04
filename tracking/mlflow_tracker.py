
import mlflow


class MLflowTracker:

    def __init__(
        self,
        experiment_name="AutoResearch-Lab"
    ):
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)

    def log_parameters(self, parameters):
        mlflow.log_params(parameters)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path="model"):
        mlflow.pytorch.log_model(
            model,
            name=artifact_path
        )

    def end_run(self):
        mlflow.end_run()


if __name__ == "__main__":

    tracker = MLflowTracker()

    with mlflow.start_run(run_name="baseline_experiment"):

        tracker.log_parameters({
            "model": "DistilBERT",
            "dataset": "AG News",
            "learning_rate": 2e-5,
            "epochs": 3,
            "batch_size": 16
        })

        tracker.log_metrics({
            "training_loss": 0.45,
            "accuracy": 0.90,
            "f1_score": 0.89
        })

    print("MLflow experiment logged successfully.")
