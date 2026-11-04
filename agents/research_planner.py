
from dataclasses import dataclass
from typing import List


@dataclass
class ResearchPlan:
    research_problem: str
    hypothesis: str
    objectives: List[str]
    evaluation_metrics: List[str]


class ResearchPlanner:

    def create_plan(self, research_problem: str) -> ResearchPlan:
        if not research_problem.strip():
            raise ValueError("Research problem cannot be empty.")

        hypothesis = (
            "Systematic experimentation can identify "
            "measurable improvements in model performance."
        )

        objectives = [
            "Establish a baseline model",
            "Design controlled experiments",
            "Evaluate model performance",
            "Analyze experimental results",
            "Document research findings"
        ]

        evaluation_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "training_time",
            "inference_latency"
        ]

        return ResearchPlan(
            research_problem=research_problem,
            hypothesis=hypothesis,
            objectives=objectives,
            evaluation_metrics=evaluation_metrics
        )


if __name__ == "__main__":
    planner = ResearchPlanner()

    plan = planner.create_plan(
        "Investigating efficient Transformer architectures"
    )

    print(plan)
