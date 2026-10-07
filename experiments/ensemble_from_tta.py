"""
Step 3c: Combine already-computed TTA probability arrays (from
experiments/tta_evaluate.py's saved .npy outputs) via weighted soft voting
-- no forward passes at all here, just loading arrays and averaging, so
this runs in under a second.

Usage:
    python experiments/ensemble_from_tta.py --config configs/config.yaml --seed 42 \
        --models classical_proposed hybrid --representations cnn fusion \
        --weights 0.5 0.5

Place this at: experiments/ensemble_from_tta.py
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Ensemble over saved TTA probability arrays")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="+", required=True,
                         help="model names as passed to tta_evaluate.py's --model, e.g. classical_proposed hybrid")
    parser.add_argument("--representations", type=str, nargs="+", required=True,
                         help="matching representation for each --models entry, e.g. cnn fusion")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                         help="defaults to equal weighting; normalized automatically")
    parser.add_argument("--name", type=str, default="ensemble_tta")
    args = parser.parse_args()

    if len(args.models) != len(args.representations):
        raise ValueError("--models and --representations must have the same length (one representation per model)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    results_dir = Path(cfg["project"]["results_dir"])

    run_names = []
    probs_list = []
    labels_ref = None
    for model_name, representation in zip(args.models, args.representations):
        run_prefix = "" if representation == "cnn" else f"{representation}_"
        run_name = f"{run_prefix}{model_name}"
        run_names.append(run_name)

        probs_path = results_dir / "logs" / f"tta_probs_{run_name}_seed{seed}.npy"
        labels_path = results_dir / "logs" / f"tta_labels_{run_name}_seed{seed}.npy"
        if not probs_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Missing saved TTA arrays for '{run_name}': {probs_path}\n"
                f"Run first: python experiments/tta_evaluate.py --model {model_name} "
                f"--representation {representation} --config {args.config} --seed {seed}"
            )
        probs = np.load(probs_path)
        labels = np.load(labels_path)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels, labels_ref):
            raise RuntimeError(
                f"Label mismatch for '{run_name}' vs. the reference member -- these TTA runs were "
                f"not evaluated over the same samples in the same order. Aborting rather than "
                f"producing a bogus combined result."
            )
        probs_list.append(probs)
        print(f"  Loaded TTA probs for '{run_name}'")

    weights = args.weights or [1.0 / len(run_names)] * len(run_names)
    if len(weights) != len(run_names):
        raise ValueError(f"Got {len(weights)} weights for {len(run_names)} models")
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    combined_probs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        combined_probs += w * p

    y_pred = combined_probs.argmax(axis=1)
    n_classes = combined_probs.shape[1]
    metrics = compute_metrics(labels_ref, y_pred, combined_probs, n_classes=n_classes)

    print(f"\nTTA-ensemble members: {list(zip(run_names, weights))}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1_macro: {metrics['f1_macro']:.4f}")
    print("confusion_matrix:")
    print(np.array(metrics["confusion_matrix"]))

    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    eval_log = {k: v for k, v in metrics.items() if k != "per_class_report"}
    eval_log["members"] = list(zip(run_names, weights))
    eval_log["tta"] = "horizontal_flip (per member, pre-combined)"
    with open(results_dir / "logs" / f"eval_{args.name}_seed{seed}.json", "w") as f:
        json.dump(eval_log, f, indent=2)

    table_path = results_dir / "tables" / "ensemble_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ensemble_name": args.name, "members": "+".join(run_names),
        "weights": ",".join(f"{w:.4f}" for w in weights), "seed": seed,
        "accuracy": metrics["accuracy"], "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"], "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"], "roc_auc_ovr": metrics.get("roc_auc_ovr"),
        "specificity_macro": metrics.get("specificity_macro"),
        "git_commit": _get_git_commit(),
    }
    write_header = not table_path.exists()
    with open(table_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved to {table_path}")


if __name__ == "__main__":
    main()