
from dataclasses import dataclass
from typing import Callable, Dict, List, Any


@dataclass
class AblationResult:
    experiment_name: str
    configuration: Dict[str, Any]
    metrics: Dict[str, float]


class AblationStudy:

    def __init__(self, baseline_metrics):
        self.baseline_metrics = baseline_metrics
        self.results: List[AblationResult] = []

    def run_experiment(
        self,
        experiment_name: str,
        configuration: Dict[str, Any],
        experiment_fn: Callable
    ):
        metrics = experiment_fn(configuration)

        if not isinstance(metrics, dict):
            raise TypeError("Experiment must return a metrics dictionary.")

        result = AblationResult(
            experiment_name=experiment_name,
            configuration=configuration,
            metrics=metrics
        )

        self.results.append(result)

        return result

    def compare_results(self, metric="accuracy"):

        if metric not in self.baseline_metrics:
            raise ValueError("Metric not found in baseline.")

        baseline = self.baseline_metrics[metric]

        comparisons = []

        for result in self.results:

            if metric not in result.metrics:
                continue

            score = result.metrics[metric]

            comparisons.append({
                "experiment": result.experiment_name,
                "baseline_score": baseline,
                "experiment_score": score,
                "difference": score - baseline
            })

        return comparisons


if __name__ == "__main__":

    baseline = {
        "accuracy": 0.85,
        "f1_score": 0.84
    }

    study = AblationStudy(baseline)

    # Demonstration experiment function
    def mock_experiment(configuration):
        return {
            "accuracy": 0.87,
            "f1_score": 0.86
        }

    study.run_experiment(
        experiment_name="attention_ablation",
        configuration={
            "attention": "modified",
            "dropout": 0.1
        },
        experiment_fn=mock_experiment
    )

    results = study.compare_results("accuracy")

    print("Ablation Study Results")

    for result in results:
        print(result)
