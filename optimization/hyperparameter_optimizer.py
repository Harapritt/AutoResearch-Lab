import optuna
from dataclasses import dataclass
from typing import Callable, Dict, Any


@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_value: float
    number_of_trials: int


class HyperparameterOptimizer:
    def __init__(self, direction: str = "maximize"):
        if direction not in {"maximize", "minimize"}:
            raise ValueError("Direction must be 'maximize' or 'minimize'.")

        self.direction = direction

    def optimize(
        self,
        objective_fn: Callable,
        n_trials: int = 20
    ) -> OptimizationResult:

        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")

        study = optuna.create_study(
            direction=self.direction
        )

        study.optimize(
            objective_fn,
            n_trials=n_trials
        )

        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            number_of_trials=len(study.trials)
        )


def objective(trial):
    """
    Demonstration objective function.

    Replace this with a real model-training and
    validation process before using the optimizer
    for research results.
    """

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        5e-4,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [8, 16, 32]
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True
    )

    # Placeholder score for testing the Optuna workflow.
    score = (
        1.0
        - abs(learning_rate - 1e-4) * 1000
        - abs(batch_size - 16) * 0.002
        - weight_decay
    )

    return score


if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()

    result = optimizer.optimize(
        objective_fn=objective,
        n_trials=20
    )

    print("Best parameters:", result.best_params)
    print("Best score:", result.best_value)
    print("Total trials:", result.number_of_trials)