import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ExperimentLogger:

    def __init__(self, log_path: str = "results/experiments.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.fields = [
            "experiment_name",
            "timestamp",
            "parameters",
            "metrics",
            "duration_seconds"
        ]

        if not self.log_path.exists():
            with self.log_path.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fields
                )
                writer.writeheader()

    def log_experiment(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        duration_seconds: float
    ) -> None:

        if duration_seconds < 0:
            raise ValueError("Duration cannot be negative.")

        row = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": str(parameters),
            "metrics": str(metrics),
            "duration_seconds": duration_seconds
        }

        with self.log_path.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fields
            )
            writer.writerow(row)