import numpy as np
from scipy.stats import wilcoxon
from typing import Tuple

def wilcoxon_signed_rank(x, y):
    """
    Perform Wilcoxon signed-rank test for paired samples.
    Args:
        x, y: Arrays of paired observations (e.g., accuracy with and without DP)
    Returns:
        statistic, p-value
    """
    res = wilcoxon(x, y)
    return float(res.statistic), float(res.pvalue) # type: ignore   

def confidence_interval(data, confidence=0.95) -> Tuple[float, float]:
    """
    Compute the confidence interval for the mean of the data.
    Args:
        data: List or np.array of values
        confidence: Confidence level (default 0.95)
    Returns:
        (mean, half-width of interval)
    """
    data = np.array(data)
    n = len(data)
    mean_val = float(np.mean(data))
    se = float(np.std(data, ddof=1)) / np.sqrt(n)
    from scipy.stats import t
    h = float(se * t.ppf((1 + confidence) / 2., n-1))
    return mean_val, h

def print_statistical_results(metric_name, with_privacy, without_privacy):
    """
    Print statistical comparison results for a metric with and without privacy.
    Args:
        metric_name: Name of the metric (e.g., 'Accuracy')
        with_privacy: List/array of metric values with privacy enabled
        without_privacy: List/array of metric values without privacy
    """
    stat, p = wilcoxon_signed_rank(with_privacy, without_privacy)
    mean_priv, ci_priv = confidence_interval(with_privacy)
    mean_no_priv, ci_no_priv = confidence_interval(without_privacy)
    print(f"{metric_name} with privacy: {mean_priv:.4f} ± {ci_priv:.4f}")
    print(f"{metric_name} without privacy: {mean_no_priv:.4f} ± {ci_no_priv:.4f}")
    print(f"Wilcoxon signed-rank test: statistic={stat:.4f}, p-value={p:.4g}")
    if p < 0.05:
        print("Difference is statistically significant (p < 0.05).")
    else:
        print("No statistically significant difference (p >= 0.05).")

# Optional: Privacy leakage metric stub (for future extension)
def privacy_leakage_metric(*args, **kwargs):
    """
    Placeholder for privacy leakage quantification (e.g., membership inference attack success rate).
    Extend as needed for your experiments.
    """
    raise NotImplementedError("Privacy leakage metric not implemented yet.")

# Example usage:
# acc_with_dp = [0.85, 0.86, 0.87]
# acc_without_dp = [0.88, 0.89, 0.90]
# print_statistical_results("Accuracy", acc_with_dp, acc_without_dp)
