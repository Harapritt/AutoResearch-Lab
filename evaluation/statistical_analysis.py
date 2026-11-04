
import numpy as np
from scipy import stats


class StatisticalAnalyzer:

    @staticmethod
    def compare_experiments(
        baseline_scores,
        improved_scores,
        alpha=0.05
    ):
        baseline_scores = np.asarray(
            baseline_scores, dtype=float
        )

        improved_scores = np.asarray(
            improved_scores, dtype=float
        )

        if baseline_scores.ndim != 1 or improved_scores.ndim != 1:
            raise ValueError("Scores must be one-dimensional.")

        if len(baseline_scores) != len(improved_scores):
            raise ValueError(
                "Both experiments must have the same number of paired scores."
            )

        if len(baseline_scores) < 2:
            raise ValueError(
                "At least two paired observations are required."
            )

        if not np.all(np.isfinite(baseline_scores)):
            raise ValueError("Baseline scores must be finite.")

        if not np.all(np.isfinite(improved_scores)):
            raise ValueError("Improved scores must be finite.")

        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")

        # Paired statistical comparison
        test_result = stats.ttest_rel(
            improved_scores,
            baseline_scores
        )

        baseline_mean = np.mean(baseline_scores)
        improved_mean = np.mean(improved_scores)

        improvement = improved_mean - baseline_mean

        p_value = float(test_result.pvalue)

        return {
            "baseline_mean": float(baseline_mean),
            "improved_mean": float(improved_mean),
            "mean_improvement": float(improvement),
            "p_value": p_value,
            "statistically_significant": bool(
                p_value < alpha
            ),
            "significance_level": alpha
        }


if __name__ == "__main__":

    baseline = [0.84, 0.85, 0.83, 0.86, 0.84]

    improved = [0.87, 0.88, 0.86, 0.89, 0.87]

    analyzer = StatisticalAnalyzer()

    results = analyzer.compare_experiments(
        baseline,
        improved
    )

    print("Statistical Analysis Results")

    for metric, value in results.items():
        print(f"{metric}: {value}")
