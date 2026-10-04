"""
Step 3: Soft-vote ensemble over already-trained checkpoints.

No retraining -- loads existing checkpoints for classical_proposed, hybrid
(cnn representation), and fusion_hybrid (fusion representation), runs each
on the test set, and combines their softmax probability outputs via a
weighted average. gnn_hybrid is excluded by default (pass --include_gnn to
add it) since its current ~40% accuracy would drag the ensemble down rather
than help it -- an ensemble member only helps if its errors are usefully
uncorrelated with the others', not just "different because it's weak".

CORRECTNESS NOTE: classical_proposed/hybrid use one dataloader (plain
images, build_dataloaders) and fusion_hybrid uses a different one
(images + graphs, build_fusion_dataloaders). Both are built from the same
config/seed so patient_level_split should produce identical sample order,
but we do NOT just trust that -- every batch asserts the labels from both
loaders match before combining predictions, and the script aborts loudly
if they ever don't, rather than silently averaging predictions for
different images.

Usage:
    python experiments/ensemble.py --config configs/config.yaml --seed 42
    python experiments/ensemble.py --config configs/config.yaml --seed 42 \
        --weights 0.33 0.33 0.34   # classical_proposed, hybrid, fusion_hybrid order
    python experiments/ensemble.py --config configs/config.yaml --seed 42 --include_gnn
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import build_dataloaders
from data.fusion_dataset import build_fusion_dataloaders
from models.graph import build_graph_dataloaders
from train import build_model, _forward_batch
from utils.metrics import compute_metrics
from utils.reproducibility import set_seed, get_device

ENSEMBLE_MEMBERS = ["classical_proposed", "hybrid", "fusion_hybrid"]
ENSEMBLE_RUN_SPECS = {
    # name -> (model_name for build_model, representation, checkpoint_prefix)
    "classical_proposed": ("classical_proposed", "cnn", ""),
    "hybrid": ("hybrid", "cnn", ""),
    "fusion_hybrid": ("hybrid", "fusion", "fusion_"),
    "gnn_hybrid": ("hybrid", "gnn", "gnn_"),
}


def load_member(name: str, cfg: dict, n_classes: int, seed: int, device):
    model_name, representation, prefix = ENSEMBLE_RUN_SPECS[name]
    model, _ = build_model(model_name, cfg, n_classes, representation)
    model.to(device)

    checkpoint_path = Path(cfg["project"]["results_dir"]) / "checkpoints" / f"{prefix}{model_name}_seed{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for ensemble member '{name}': {checkpoint_path}\n"
            f"Train it first with: python train.py --model {model_name} "
            f"--representation {representation} --seed {seed}"
        )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, representation


@torch.no_grad()
def collect_member_probs(model, loader, device, representation: str):
    """Returns (probs: (N, n_classes) ndarray, labels: (N,) ndarray), in loader order."""
    all_probs, all_labels = [], []
    for batch in loader:
        logits, y = _forward_batch(model, batch, device, representation)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def run_ensemble(cfg: dict, seed: int, member_names: list, weights: list, output_prefix: str):
    set_seed(seed)
    device = get_device(cfg["project"]["device"])

    # cnn-representation members (classical_proposed, hybrid) share one test loader;
    # fusion-representation members get their own. We never trust that two
    # independently-built loaders line up without checking -- see module docstring.
    cnn_needed = any(ENSEMBLE_RUN_SPECS[n][1] == "cnn" for n in member_names)
    fusion_needed = any(ENSEMBLE_RUN_SPECS[n][1] == "fusion" for n in member_names)
    gnn_needed = any(ENSEMBLE_RUN_SPECS[n][1] == "gnn" for n in member_names)

    cnn_test_loader = fusion_test_loader = gnn_test_loader = None
    classes = None
    if cnn_needed:
        _, _, cnn_test_loader, classes, _ = build_dataloaders(cfg, seed=seed)
    if fusion_needed:
        _, _, fusion_test_loader, fusion_classes, _ = build_fusion_dataloaders(cfg)
        classes = classes or fusion_classes
    if gnn_needed:
        _, _, gnn_test_loader, gnn_classes, _ = build_graph_dataloaders(cfg)
        classes = classes or gnn_classes
    n_classes = len(classes)

    member_probs = {}
    member_labels_ref = None

    for name in member_names:
        model, representation = load_member(name, cfg, n_classes, seed, device)
        loader = {"cnn": cnn_test_loader, "fusion": fusion_test_loader, "gnn": gnn_test_loader}[representation]
        probs, labels = collect_member_probs(model, loader, device, representation)
        member_probs[name] = probs
        if member_labels_ref is None:
            member_labels_ref = labels
        else:
            if not np.array_equal(labels, member_labels_ref):
                raise RuntimeError(
                    f"Label mismatch between ensemble member '{name}' and the reference member -- "
                    f"this means the dataloaders are NOT iterating over the same samples in the "
                    f"same order, and combining their predictions would silently be wrong. "
                    f"Aborting rather than producing a bogus ensemble result."
                )
        print(f"  Loaded predictions for '{name}' (representation={representation})")

    if weights is None:
        weights = [1.0 / len(member_names)] * len(member_names)
    if len(weights) != len(member_names):
        raise ValueError(f"Got {len(weights)} weights for {len(member_names)} members")
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]  # normalize

    ensemble_probs = np.zeros_like(member_probs[member_names[0]])
    for name, w in zip(member_names, weights):
        ensemble_probs += w * member_probs[name]

    y_pred = ensemble_probs.argmax(axis=1)
    y_true = member_labels_ref

    metrics = compute_metrics(y_true, y_pred, ensemble_probs, n_classes=n_classes)

    print(f"\nEnsemble members: {list(zip(member_names, weights))}")
    print(f"Ensemble accuracy: {metrics['accuracy']:.4f}")
    print(f"Ensemble f1_macro: {metrics['f1_macro']:.4f}")
    print("Confusion matrix:")
    print(classes)
    print(np.array(metrics["confusion_matrix"]))

    results_dir = Path(cfg["project"]["results_dir"])
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    eval_log = {k: v for k, v in metrics.items() if k != "per_class_report"}
    eval_log["classes"] = classes
    eval_log["members"] = list(zip(member_names, weights))
    with open(results_dir / "logs" / f"eval_{output_prefix}_seed{seed}.json", "w") as f:
        json.dump(eval_log, f, indent=2)

    table_path = results_dir / "tables" / "ensemble_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ensemble_name": output_prefix,
        "members": "+".join(member_names),
        "weights": ",".join(f"{w:.4f}" for w in weights),
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "roc_auc_ovr": metrics.get("roc_auc_ovr"),
        "specificity_macro": metrics.get("specificity_macro"),
    }
    write_header = not table_path.exists()
    with open(table_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved to {table_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Soft-vote ensemble over trained checkpoints")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                         help="Weights in the same order as the member list "
                              "(default: classical_proposed hybrid fusion_hybrid [gnn_hybrid]). "
                              "Need not sum to 1 -- normalized automatically.")
    parser.add_argument("--include_gnn", action="store_true",
                         help="Also include gnn_hybrid as a 4th ensemble member (off by default -- "
                              "its current accuracy is too low to help).")
    parser.add_argument("--name", type=str, default=None, help="Output name for logs/table row")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    member_names = list(ENSEMBLE_MEMBERS)
    if args.include_gnn:
        member_names.append("gnn_hybrid")

    output_prefix = args.name or ("ensemble_" + "_".join(member_names))

    run_ensemble(cfg, seed, member_names, args.weights, output_prefix)


if __name__ == "__main__":
    main()