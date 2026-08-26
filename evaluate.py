"""
Evaluates a trained model on the held-out test set.

The model is loaded from an existing checkpoint produced by train.py.
No training is performed during evaluation.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.dataset import build_dataloaders
from train import build_model
from utils.metrics import compute_metrics
from utils.reproducibility import set_seed, get_device
from utils.param_count import build_param_report


RESULTS_TABLE_COLUMNS = [
    "model",
    "dataset_split_method",
    "seed",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc_ovr",
    "specificity_macro",
    "total_params",
    "quantum_parameters",
    "qubits",
    "circuit_depth",
    "training_time_sec",
    "inference_time_sec",
]


@torch.no_grad()
def evaluate_model(model, test_loader, device, n_classes):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    t0 = time.time()

    for x, y in test_loader:
        x = x.to(device)

        logits = model(x)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())

    inference_time = time.time() - t0

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(
        y_true,
        y_pred,
        y_prob,
        n_classes=n_classes,
    )

    metrics["inference_time_sec"] = inference_time
    metrics["n_test_samples"] = len(y_true)

    return metrics, y_true, y_pred, y_prob


def append_to_results_table(row: dict, table_path: str):
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not table_path.exists()

    with open(table_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULTS_TABLE_COLUMNS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                k: row.get(k)
                for k in RESULTS_TABLE_COLUMNS
            }
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "simple_cnn",
            "resnet18",
            "mobilenet_v2",
            "classical_proposed",
            "hybrid",
        ],
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = (
        args.seed
        if args.seed is not None
        else cfg["project"]["seed"]
    )

    set_seed(seed)

    device = get_device(
        cfg["project"]["device"]
    )

    # ---------------------------------------------------------
    # Build data loaders
    # ---------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        classes,
        meta,
    ) = build_dataloaders(cfg)

    n_classes = len(classes)

    # ---------------------------------------------------------
    # Build model architecture
    # ---------------------------------------------------------

    model, is_quantum = build_model(
        args.model,
        cfg,
        n_classes,
    )

    model.to(device)

    # ---------------------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------------------

    results_dir = Path(
        cfg["project"]["results_dir"]
    )

    checkpoint_path = (
        results_dir
        / "checkpoints"
        / f"{args.model}_seed{seed}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first using:\n"
            f"python train.py --model {args.model} --seed {seed}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    # ---------------------------------------------------------
    # Parameter accounting
    # ---------------------------------------------------------

    if is_quantum:

        q = cfg["quantum"]

        param_report = build_param_report(
            args.model,
            model,
            model.qlayer.quantum_parameters,
            n_qubits=q["n_qubits"],
            n_layers=q["n_layers"],
            entanglement=q["entanglement"],
            data_reuploading=q["data_reuploading"],
        )

    else:

        param_report = build_param_report(
            args.model,
            model,
            None,
            n_qubits=0,
            n_layers=0,
            entanglement="none",
            data_reuploading=False,
        )

    # ---------------------------------------------------------
    # Load training log
    # ---------------------------------------------------------

    train_log_path = (
        results_dir
        / "logs"
        / f"train_{args.model}_seed{seed}.json"
    )

    if train_log_path.exists():

        with open(train_log_path) as f:
            train_log = json.load(f)

        training_time = train_log.get(
            "training_time_sec",
            None,
        )

        split_method = train_log.get(
            "split_method",
            meta["split_method"],
        )

    else:

        training_time = None
        split_method = meta["split_method"]

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    metrics, y_true, y_pred, y_prob = evaluate_model(
        model,
        test_loader,
        device,
        n_classes,
    )

    # ---------------------------------------------------------
    # Save evaluation log
    # ---------------------------------------------------------

    (results_dir / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_log = {
        k: v
        for k, v in metrics.items()
        if k != "per_class_report"
    }

    eval_log["classes"] = classes
    eval_log["checkpoint"] = str(
        checkpoint_path
    )

    with open(
        results_dir
        / "logs"
        / f"eval_{args.model}_seed{seed}.json",
        "w",
    ) as f:

        json.dump(
            eval_log,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Experiment tracking
    # ---------------------------------------------------------

    row = {

        "model": args.model,

        "dataset_split_method": split_method,

        "seed": seed,

        "accuracy": metrics["accuracy"],

        "precision_macro": metrics["precision_macro"],

        "recall_macro": metrics["recall_macro"],

        "f1_macro": metrics["f1_macro"],

        "f1_weighted": metrics["f1_weighted"],

        "roc_auc_ovr": metrics.get(
            "roc_auc_ovr"
        ),

        "specificity_macro": metrics.get(
            "specificity_macro"
        ),

        "total_params": param_report.total_params,

        "quantum_parameters": param_report.quantum_params,

        "qubits": param_report.n_qubits,

        "circuit_depth": param_report.circuit_depth,

        "training_time_sec": training_time,

        "inference_time_sec": metrics[
            "inference_time_sec"
        ],
    }

    append_to_results_table(
        row,
        results_dir
        / "tables"
        / "experiment_results.csv",
    )

    print(
        json.dumps(
            row,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()