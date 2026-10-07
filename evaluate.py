"""
Evaluates a trained model on the held-out test set.

The model is loaded from an existing checkpoint produced by train.py.
No training is performed during evaluation.

Step 2 changes:
  - --representation now accepts "fusion" in addition to "cnn"/"gnn".
  - Checkpoint/log path naming generalized to match train.py's Step 2 change:
    only "cnn" has no prefix; "gnn" and "fusion" each get their own prefix
    (representation + "_" + model_name).
  - Batch forward pass now reuses train.py's _forward_batch() instead of a
    separate hasattr(batch, "edge_index") duck-typing check, so fusion's
    (images, graph_batch, labels) tuple batches are handled correctly.
  - build_external_test_loader (used by --test_root, i.e. cross-dataset eval)
    does NOT yet support representation="fusion" -- it raises a clear error
    rather than silently mishandling the batch shape. Cross-dataset fusion
    support is deferred to the step where we actually run cross-dataset
    experiments on the fusion model.
"""

import argparse
import csv
import json
import subprocess
import time
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
import torch
import torch.nn.functional as F
import yaml

from data.dataset import MRIDataset, build_dataloaders, build_transforms, collect_samples
from data.fusion_dataset import build_fusion_dataloaders
from models.graph import GraphMRIDataset, build_graph_dataloaders
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from train import build_model, _forward_batch
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
    "git_commit",
]


def build_external_test_loader(cfg: dict, target_root: str, representation: str):
    """Build an unsplit test loader for an external ImageFolder dataset."""
    if representation == "fusion":
        raise NotImplementedError(
            "External/cross-dataset evaluation for representation='fusion' is not yet implemented "
            "(build_fusion_dataloaders only builds the configured train/val/test split from "
            "cfg['data']['root'], it has no external-root variant yet)."
        )

    samples, classes = collect_samples(target_root)
    expected_classes = sorted(classes)
    data_cfg = cfg["data"]
    transform = build_transforms(data_cfg["image_size"], data_cfg["augmentation"], train=False,
                                  crop_margin=data_cfg.get("margin_crop", True))

    if representation == "gnn":
        graph_cfg = cfg.get("graph", {})
        graph_cfg = {key: graph_cfg[key] for key in ("n_segments", "compactness") if key in graph_cfg}
        dataset = GraphMRIDataset(samples, transform, graph_cfg, cfg["project"]["seed"])
    else:
        dataset = MRIDataset(samples, transform=transform)

    loader_class = GraphDataLoader if representation == "gnn" else DataLoader
    loader = loader_class(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    return loader, expected_classes


@torch.no_grad()
def evaluate_model(model, test_loader, device, n_classes, representation: str = "cnn"):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    t0 = time.time()

    for batch in test_loader:
        logits, y = _forward_batch(model, batch, device, representation)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
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
        writer = csv.DictWriter(f, fieldnames=RESULTS_TABLE_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow({k: row.get(k) for k in RESULTS_TABLE_COLUMNS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["simple_cnn", "resnet18", "mobilenet_v2", "classical_proposed", "hybrid"],
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--representation", choices=["cnn", "gnn", "fusion"], default="cnn")
    parser.add_argument("--test_root", type=str, default=None,
                         help="Optional external ImageFolder root to evaluate without splitting")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(cfg["project"]["device"])

    # ---------------------------------------------------------
    # Build data loaders
    # ---------------------------------------------------------
    if args.test_root:
        test_loader, classes = build_external_test_loader(cfg, args.test_root, args.representation)
        meta = {"split_method": "external_full_dataset"}
        train_loader = val_loader = None
    else:
        if args.representation == "gnn":
            train_loader, val_loader, test_loader, classes, meta = build_graph_dataloaders(cfg)
        elif args.representation == "fusion":
            train_loader, val_loader, test_loader, classes, meta = build_fusion_dataloaders(cfg)
        else:
            train_loader, val_loader, test_loader, classes, meta = build_dataloaders(cfg, seed=seed)

    n_classes = len(classes)

    # ---------------------------------------------------------
    # Build model architecture
    # ---------------------------------------------------------
    model, is_quantum = build_model(args.model, cfg, n_classes, args.representation)
    model.to(device)

    if args.test_root and classes != sorted(classes):
        raise ValueError("External dataset classes must be sorted consistently")

    # ---------------------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------------------
    results_dir = Path(cfg["project"]["results_dir"])
    # Same generalized naming as train.py: only "cnn" has no prefix.
    run_prefix = "" if args.representation == "cnn" else f"{args.representation}_"
    checkpoint_path = results_dir / "checkpoints" / f"{run_prefix}{args.model}_seed{seed}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first using:\n"
            f"python train.py --model {args.model} --representation {args.representation} --seed {seed}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # ---------------------------------------------------------
    # Parameter accounting
    # ---------------------------------------------------------
    run_name = f"{run_prefix}{args.model}"
    if is_quantum:
        q = cfg["quantum"]
        param_report = build_param_report(
            run_name, model, model.qlayer.quantum_parameters,
            n_qubits=q["n_qubits"], n_layers=q["n_layers"],
            entanglement=q["entanglement"], data_reuploading=q["data_reuploading"],
        )
    else:
        param_report = build_param_report(
            run_name, model, None, n_qubits=0, n_layers=0, entanglement="none", data_reuploading=False,
        )

    # ---------------------------------------------------------
    # Load training log
    # ---------------------------------------------------------
    train_log_path = results_dir / "logs" / f"train_{run_prefix}{args.model}_seed{seed}.json"

    if train_log_path.exists():
        with open(train_log_path) as f:
            train_log = json.load(f)
        training_time = train_log.get("training_time_sec", None)
        split_method = meta["split_method"] if args.test_root else train_log.get("split_method", meta["split_method"])
    else:
        training_time = None
        split_method = meta["split_method"]

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, n_classes, args.representation)

    # ---------------------------------------------------------
    # Save evaluation log
    # ---------------------------------------------------------
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)

    eval_log = {k: v for k, v in metrics.items() if k != "per_class_report"}
    eval_log["classes"] = classes
    eval_log["checkpoint"] = str(checkpoint_path)

    with open(results_dir / "logs" / f"eval_{run_prefix}{args.model}_seed{seed}.json", "w") as f:
        json.dump(eval_log, f, indent=2)

    # ---------------------------------------------------------
    # Experiment tracking
    # ---------------------------------------------------------
    row = {
        "model": run_name,
        "dataset_split_method": split_method,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "roc_auc_ovr": metrics.get("roc_auc_ovr"),
        "specificity_macro": metrics.get("specificity_macro"),
        "total_params": param_report.total_params,
        "quantum_parameters": param_report.quantum_params,
        "qubits": param_report.n_qubits,
        "circuit_depth": param_report.circuit_depth,
        "training_time_sec": training_time,
        "inference_time_sec": metrics["inference_time_sec"],
        "git_commit": _get_git_commit(),
    }

    append_to_results_table(row, results_dir / "tables" / "experiment_results.csv")

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()