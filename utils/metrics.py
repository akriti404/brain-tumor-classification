"""Multi-class evaluation metrics beyond raw accuracy (spec Section 10)."""
import numpy as np
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
