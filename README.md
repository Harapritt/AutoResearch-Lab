AutoResearch-Lab

An autonomous machine learning research and experimentation framework for model training, hyperparameter optimization, ablation studies, statistical evaluation, and reproducible experiments.

About

AutoResearch-Lab is a research-oriented machine learning project designed to organize and automate key stages of model experimentation.

The project uses the AG News text classification dataset and Transformer-based models to explore model performance and experimental configurations.

Key Features

- Experiment Configuration: Centralized configuration for training parameters.
- Model Training: Integrated training and evaluation pipeline.
- Hyperparameter Optimization: Optuna-based optimization of learning rate and batch size.
- Ablation Studies: Comparison of different sequence lengths.
- Evaluation: Accuracy, precision, recall, F1-score, and classification reports.
- Statistical Analysis: Statistical comparison of experimental results.
- Experiment Tracking: MLflow-based experiment logging.
- Reproducibility: Random seed management and device selection.
- Testing: Automated tests for core project components.

Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Optuna
- MLflow
- NumPy
- Pandas
- SciPy
- Matplotlib
- PyYAML
- Pytest

Dataset

The project uses the AG News dataset, a four-class news topic classification dataset.

The dataset is loaded through the Hugging Face Datasets library.

Project Workflow

1. Configure the experiment.
2. Load and preprocess the dataset.
3. Initialize the Transformer model.
4. Train the model.
5. Evaluate model performance.
6. Optimize hyperparameters.
7. Run ablation experiments.
8. Analyze experimental results.
9. Track experiments and maintain reproducibility.

Running the Project

Install dependencies:

pip install -r requirements.txt

Run the integrated experiment:

python -m experiments.run_experiment

Run hyperparameter optimization:

python -m optimization.run_optimization

Run ablation experiments:

python -m experiments.run_ablation

Run tests:

python -m pytest tests/

Evaluation Metrics

The framework supports:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Statistical significance testing

Research Focus

The project focuses on reproducible machine learning experimentation, Transformer-based text classification, hyperparameter optimization, and systematic model evaluation.

Development Status

The core framework and experiment scripts have been developed. Full execution, test results, and experimental findings should be reported after successful verification.

Future Improvements

- Validation-based hyperparameter selection
- More model architectures
- Expanded datasets
- Improved experiment automation
- More comprehensive statistical comparisons

License

This project is licensed under the MIT License. See "LICENSE" (LICENSE) for details.