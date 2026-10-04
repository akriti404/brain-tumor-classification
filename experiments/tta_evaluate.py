"""
Step 3b: Test-time augmentation (TTA) -- evaluate a trained model on the
original test image AND its horizontal flip, average the two softmax
outputs, and report metrics.

For "cnn" representation (classical_proposed, hybrid): the flip is applied
directly to the already-loaded image tensor in the eval loop -- cheap,
no new dataset needed, since these models have no dependency on image
content beyond the pixels themselves.

For "fusion" representation (fusion_hybrid): flipping the image changes
what the SLIC superpixel graph SHOULD look like, so we cannot just flip
the tensor and reuse the original graph -- that would silently feed the
model a graph that describes the UN-flipped image alongside a flipped CNN
input, which is not a valid augmentation, it's just broken input. Instead
we build a second dataloader with horizontal_flip=True (see data/
fusion_dataset.py), which flips the tensor BEFORE building the graph, so
both modalities agree on which view they're looking at. We then iterate
both loaders in lockstep and assert their labels match before combining,
same safety pattern as experiments/ensemble.py.

"gnn" representation is not supported here (would need the same
flip-before-graph treatment in models/graph.py / GraphMRIDataset, which
hasn't been added -- skip gnn_hybrid for TTA since it's excluded from the
ensemble anyway).

Usage:
    python experiments/tta_evaluate.py --model classical_proposed --config configs/config.yaml --seed 42
    python experiments/tta_evaluate.py --model hybrid --representation cnn --config configs/config.yaml --seed 42
    python experiments/tta_evaluate.py --model hybrid --representation fusion --config configs/config.yaml --seed 42
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
from train import build_model
from utils.metrics import compute_metrics
from utils.reproducibility import set_seed, get_device


@torch.no_grad()
def tta_predict_cnn(model, test_loader, device):
    all_probs, all_labels = [], []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        probs_orig = F.softmax(model(images), dim=1)
        probs_flip = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
        avg_probs = (probs_orig + probs_flip) / 2
        all_probs.append(avg_probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def tta_predict_fusion(model, orig_loader, flip_loader, device):
    all_probs, all_labels = [], []
    for (img_o, graph_o, y_o), (img_f, graph_f, y_f) in zip(orig_loader, flip_loader):
        if not torch.equal(y_o, y_f):
            raise RuntimeError(
                "Label mismatch between original and flipped fusion loaders -- they are not "
                "iterating over the same samples in the same order. Aborting rather than "
                "producing a bogus TTA result."
            )
        img_o, graph_o, y_o = img_o.to(device), graph_o.to(device), y_o.to(device)
        img_f, graph_f = img_f.to(device), graph_f.to(device)

        probs_orig = F.softmax(model(img_o, graph_o), dim=1)
        probs_flip = F.softmax(model(img_f, graph_f), dim=1)
        avg_probs = (probs_orig + probs_flip) / 2
        all_probs.append(avg_probs.cpu().numpy())
        all_labels.append(y_o.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model with horizontal-flip TTA")
    parser.add_argument("--model", type=str, required=True,
                         choices=["simple_cnn", "resnet18", "mobilenet_v2", "classical_proposed", "hybrid"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--representation", choices=["cnn", "fusion"], default="cnn")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(cfg["project"]["device"])

    if args.representation == "cnn":
        _, _, test_loader, classes, _ = build_dataloaders(cfg, seed=seed)
    else:  # fusion
        _, _, orig_test_loader, classes, _ = build_fusion_dataloaders(cfg, horizontal_flip=False)
        _, _, flip_test_loader, _, _ = build_fusion_dataloaders(cfg, horizontal_flip=True)
    n_classes = len(classes)

    model, is_quantum = build_model(args.model, cfg, n_classes, args.representation)
    model.to(device)

    run_prefix = "" if args.representation == "cnn" else f"{args.representation}_"
    checkpoint_path = Path(cfg["project"]["results_dir"]) / "checkpoints" / f"{run_prefix}{args.model}_seed{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train it first: python train.py --model {args.model} --representation {args.representation} --seed {seed}"
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    if args.representation == "cnn":
        probs, labels = tta_predict_cnn(model, test_loader, device)
    else:
        probs, labels = tta_predict_fusion(model, orig_test_loader, flip_test_loader, device)

    y_pred = probs.argmax(axis=1)
    metrics = compute_metrics(labels, y_pred, probs, n_classes=n_classes)

    run_name = f"{run_prefix}{args.model}"
    print(f"\n[{run_name}] TTA (original + horizontal-flip, averaged)")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"f1_macro:  {metrics['f1_macro']:.4f}")
    print("confusion_matrix:")
    print(classes)
    print(np.array(metrics["confusion_matrix"]))

    results_dir = Path(cfg["project"]["results_dir"])
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    eval_log = {k: v for k, v in metrics.items() if k != "per_class_report"}
    eval_log["classes"] = classes
    eval_log["tta"] = "horizontal_flip"
    with open(results_dir / "logs" / f"eval_tta_{run_name}_seed{seed}.json", "w") as f:
        json.dump(eval_log, f, indent=2)

    # Also save the raw per-sample TTA'd probabilities -- lets us feed these
    # straight into experiments/ensemble.py-style combination later without
    # recomputing the forward passes.
    np.save(results_dir / "logs" / f"tta_probs_{run_name}_seed{seed}.npy", probs)
    np.save(results_dir / "logs" / f"tta_labels_{run_name}_seed{seed}.npy", labels)

    table_path = results_dir / "tables" / "tta_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": run_name, "seed": seed, "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"], "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"], "f1_weighted": metrics["f1_weighted"],
        "roc_auc_ovr": metrics.get("roc_auc_ovr"), "specificity_macro": metrics.get("specificity_macro"),
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