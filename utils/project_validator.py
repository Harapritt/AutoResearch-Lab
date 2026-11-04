import importlib
from pathlib import Path


class ProjectValidator:

    REQUIRED_PACKAGES = [
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "optuna",
        "mlflow",
        "yaml"
    ]

    REQUIRED_FILES = [
        "requirements.txt",
        "configs/experiment_config.py",
        "models/model_loader.py",
        "evaluation/metrics.py",
        "experiments/train.py",
        "experiments/evaluate.py",
        "optimization/hyperparameter_optimizer.py",
        "tracking/mlflow_tracker.py",
        "utils/reproducibility.py"
    ]

    @classmethod
    def check_dependencies(cls):
        results = {}

        for package in cls.REQUIRED_PACKAGES:
            try:
                importlib.import_module(package)
                results[package] = "Installed"
            except ImportError:
                results[package] = "Missing"

        return results

    @classmethod
    def check_files(cls):
        return {
            file: "Found" if Path(file).is_file() else "Missing"
            for file in cls.REQUIRED_FILES
        }

    @classmethod
    def validate_project(cls):
        dependencies = cls.check_dependencies()
        files = cls.check_files()

        print("\nDEPENDENCY CHECK")

        for package, status in dependencies.items():
            print(f"{package}: {status}")

        print("\nPROJECT FILE CHECK")

        for file, status in files.items():
            print(f"{file}: {status}")

        dependencies_ok = all(
            status == "Installed"
            for status in dependencies.values()
        )

        files_ok = all(
            status == "Found"
            for status in files.values()
        )

        if dependencies_ok and files_ok:
            print("\nProject checks passed.")
        else:
            print("\nProject checks found missing dependencies or files.")

        return dependencies_ok and files_ok


if __name__ == "__main__":
    ProjectValidator.validate_project()