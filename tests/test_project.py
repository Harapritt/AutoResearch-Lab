import pytest
import torch

from configs.experiment_config import ExperimentConfig
from agents.research_planner import ResearchPlanner
from evaluation.metrics import ModelEvaluator
from optimization.hyperparameter_optimizer import HyperparameterOptimizer
from utils.reproducibility import set_seed, get_device


def test_experiment_config():

    config = ExperimentConfig()

    assert config.model_name == "distilbert-base-uncased"
    assert config.dataset_name == "ag_news"
    assert config.num_epochs > 0
    assert config.batch_size > 0

    config.validate()


def test_research_planner():

    planner = ResearchPlanner()
    plan = planner.create_plan()

    assert plan is not None


def test_model_evaluator():

    evaluator = ModelEvaluator()

    actual = [0, 1, 2, 3]
    predicted = [0, 1, 2, 2]

    results = evaluator.evaluate(actual, predicted)

    assert "accuracy" in results
    assert 0 <= results["accuracy"] <= 1


def test_optimizer_initialization():

    optimizer = HyperparameterOptimizer(
        direction="maximize"
    )

    assert optimizer.direction == "maximize"


def test_invalid_optimizer_direction():

    with pytest.raises(ValueError):
        HyperparameterOptimizer(direction="invalid")


def test_reproducibility():

    set_seed(42)
    first = torch.rand(5)

    set_seed(42)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_device():

    device = get_device()

    assert device.type in ["cpu", "cuda"]