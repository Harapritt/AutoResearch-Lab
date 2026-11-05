"""Research planning for AutoResearch-Lab."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ResearchPlan:
    research_problem: str
    hypothesis: str
    objectives: List[str]
    evaluation_metrics: List[str]
    experiment_candidates: List[Dict[str, Any]]


class ResearchPlanner:

    def create_plan(
        self,
        research_problem: str = "Improve AG News classification"
    ) -> ResearchPlan:

        problem = research_problem.strip()

        if not problem:
            raise ValueError("Research problem cannot be empty.")

        hypothesis = (
            "A controlled search over learning rate, batch size, "
            "and sequence length can improve validation F1 "
            "over the baseline configuration."
        )

        objectives = [
            "Measure a reproducible baseline",
            "Compare configurations on validation data",
            "Select the best configuration",
            "Evaluate the selected model on held-out test data",
            "Save experiment history and research reports"
        ]

        candidates = [
            {
                "name": "baseline",
                "learning_rate": 2e-5,
                "batch_size": 16,
                "max_length": 256
            },
            {
                "name": "lower_learning_rate",
                "learning_rate": 1e-5,
                "batch_size": 16,
                "max_length": 256
            },
            {
                "name": "higher_learning_rate",
                "learning_rate": 5e-5,
                "batch_size": 16,
                "max_length": 256
            },
            {
                "name": "smaller_batch",
                "learning_rate": 2e-5,
                "batch_size": 8,
                "max_length": 256
            },
            {
                "name": "longer_sequence",
                "learning_rate": 2e-5,
                "batch_size": 16,
                "max_length": 384
            }
        ]

        return ResearchPlan(
            research_problem=problem,
            hypothesis=hypothesis,
            objectives=objectives,
            evaluation_metrics=[
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "validation_loss"
            ],
            experiment_candidates=candidates
        )

    @staticmethod
    def to_dict(plan: ResearchPlan):
        return asdict(plan)


if __name__ == "__main__":
    import json

    plan = ResearchPlanner().create_plan()

    print(
        json.dumps(
            ResearchPlanner.to_dict(plan),
            indent=2
        )
    )