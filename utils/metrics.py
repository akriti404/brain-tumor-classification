"""Multi-class evaluation metrics beyond raw accuracy (spec Section 10)."""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_prob=None, n_classes=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)) if n_classes else None)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class specificity: TN / (TN + FP) from the one-vs-rest confusion matrix
    if n_classes:
        specificities = []
        for c in range(n_classes):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics["specificity_macro"] = float(np.mean(specificities))
        metrics["specificity_per_class"] = specificities

    if y_prob is not None and n_classes and n_classes > 1:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            metrics["roc_auc_ovr"] = None  # e.g. a class missing from a small smoke-test split

    metrics["per_class_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return metrics


def aggregate_over_seeds(metric_dicts: list) -> dict:
    """Given metrics from multiple seeds, return mean +/- std for scalar metrics."""
    scalar_keys = [k for k, v in metric_dicts[0].items() if isinstance(v, (int, float)) and v is not None]
    agg = {}
    for k in scalar_keys:
        vals = [m[k] for m in metric_dicts if m.get(k) is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    return agg


def compute_confidence_interval(values: list, confidence: float = 0.95) -> dict:
    """Compute confidence interval for a list of values."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    
    # Use t-distribution for small samples
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return {
        "mean": float(mean),
        "std_err": float(std_err),
        "ci_lower": float(mean - h),
        "ci_upper": float(mean + h),
        "confidence": confidence,
        "n": n
    }


def paired_t_test(group1: list, group2: list) -> dict:
    """Perform paired t-test between two groups of values."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) != len(group2):
        raise ValueError("Groups must have equal length for paired t-test")
    
    statistic, p_value = stats.ttest_rel(group1, group2)
    
    return {
        "test": "paired_t_test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_pairs": len(group1),
        "mean_diff": float(np.mean(group1 - group2)),
        "std_diff": float(np.std(group1 - group2))
    }


def wilcoxon_signed_rank_test(group1: list, group2: list) -> dict:
    """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) != len(group2):
        raise ValueError("Groups must have equal length for Wilcoxon test")
    
    statistic, p_value = stats.wilcoxon(group1, group2)
    
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_pairs": len(group1)
    }


def cohens_d(group1: list, group2: list) -> dict:
    """Compute Cohen's d effect size between two groups."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Effect size interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        "effect_size": "cohens_d",
        "value": float(d),
        "interpretation": interpretation,
        "mean1": float(mean1),
        "mean2": float(mean2),
        "pooled_std": float(pooled_std),
        "n1": n1,
        "n2": n2
    }


def compare_groups(
    group1_metrics: list,
    group2_metrics: list,
    metric_key: str = "accuracy",
    confidence: float = 0.95
) -> dict:
    """
    Comprehensive statistical comparison between two groups of metrics.
    
    Args:
        group1_metrics: List of metric dictionaries from group 1
        group2_metrics: List of metric dictionaries from group 2
        metric_key: Key to compare within the metric dictionaries
        confidence: Confidence level for confidence intervals
    
    Returns:
        Dictionary with comprehensive statistical comparison results
    """
    # Extract values for the specified metric
    group1_values = [m[metric_key] for m in group1_metrics if metric_key in m and m[metric_key] is not None]
    group2_values = [m[metric_key] for m in group2_metrics if metric_key in m and m[metric_key] is not None]
    
    if len(group1_values) < 2 or len(group2_values) < 2:
        raise ValueError("Each group must have at least 2 valid values for statistical comparison")
    
    results = {
        "metric": metric_key,
        "group1": {
            "values": group1_values,
            "n": len(group1_values),
            **compute_confidence_interval(group1_values, confidence)
        },
        "group2": {
            "values": group2_values,
            "n": len(group2_values),
            **compute_confidence_interval(group2_values, confidence)
        }
    }
    
    # If groups have equal length, perform paired tests
    if len(group1_values) == len(group2_values):
        try:
            results["paired_t_test"] = paired_t_test(group1_values, group2_values)
            results["wilcoxon_test"] = wilcoxon_signed_rank_test(group1_values, group2_values)
        except Exception as e:
            results["paired_tests_error"] = str(e)
    
    # Independent samples t-test
    try:
        statistic, p_value = stats.ttest_ind(group1_values, group2_values)
        results["independent_t_test"] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
    except Exception as e:
        results["independent_t_test_error"] = str(e)
    
    # Effect size
    try:
        results["effect_size"] = cohens_d(group1_values, group2_values)
    except Exception as e:
        results["effect_size_error"] = str(e)
    
    return results
