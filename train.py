"""
Trains one model (a classical baseline or the proposed hybrid CNN+VQC) as
specified by --model, using the shared data pipeline and config.

Usage:
    python train.py --model hybrid --config configs/config.yaml
    python train.py --model resnet18 --config configs/config.yaml
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.dataset import build_dataloaders
from models.classical import build_baseline, ClassicalProposedHead
from models.graph import build_graph_dataloaders
from models.hybrid import build_model_from_config
from models.gnn_hybrid import build_gnn_model_from_config
from utils.reproducibility import set_seed, get_device
from utils.param_count import build_param_report


def build_model(model_name: str, cfg: dict, n_classes: int, representation: str = "cnn"):
    if representation == "gnn":
        if model_name != "hybrid":
            raise ValueError("The gnn representation is only available for --model hybrid")
        return build_gnn_model_from_config(cfg, n_classes), True
    if model_name == "hybrid":
        return build_model_from_config(cfg, n_classes), True  # (model, is_quantum)
    if model_name == "classical_proposed":
        cb = cfg["classical_backbone"]
        model = ClassicalProposedHead(
            n_classes=n_classes, backbone_arch=cb["architecture"], pretrained=cb["pretrained"],
            freeze_backbone=cb["freeze_backbone"], reduced_dim=cb["reduced_dim"],
        )
        return model, False
    if model_name in ("simple_cnn", "resnet18", "mobilenet_v2"):
        model = build_baseline(model_name, n_classes, pretrained=cfg["classical_backbone"]["pretrained"])
        return model, False
    raise ValueError(f"Unknown model '{model_name}'")


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if hasattr(batch, "edge_index"):
                inputs = batch.to(device)
                y = inputs.y
            else:
                inputs, y = batch[0].to(device), batch[1].to(device)
            if train:
                optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(model_name: str, cfg: dict, seed: int = None, representation: str = "cnn"):
    seed = seed if seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(cfg["project"]["device"])

    loaders = build_graph_dataloaders(cfg) if representation == "gnn" else build_dataloaders(cfg)
    train_loader, val_loader, test_loader, classes, meta = loaders
    n_classes = len(classes)

    model, is_quantum = build_model(model_name, cfg, n_classes, representation)
    model.to(device)

    class_weights = meta["class_weights"]
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    t0 = time.time()

    for epoch in range(cfg["training"]["epochs"]):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{model_name}] epoch {epoch+1}/{cfg['training']['epochs']} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    training_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    run_name = f"{representation}_{model_name}" if representation == "gnn" else model_name

    # Parameter accounting
    if is_quantum:
        q = cfg["quantum"]
        param_report = build_param_report(
            run_name, model, model.qlayer.quantum_parameters,
            n_qubits=q["n_qubits"], n_layers=q["n_layers"],
            entanglement=q["entanglement"], data_reuploading=q["data_reuploading"],
        )
    else:
        param_report = build_param_report(model_name, model, None, n_qubits=0, n_layers=0,
                                           entanglement="none", data_reuploading=False)

    results_dir = Path(cfg["project"]["results_dir"])
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    if cfg["logging"]["save_checkpoints"]:
        (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), results_dir / "checkpoints" / f"{run_name}_seed{seed}.pt")

    log = {
        "model_name": run_name,
        "representation": representation,
        "seed": seed,
        "classes": classes,
        "split_method": meta["split_method"],
        "n_train": meta["n_train"], "n_val": meta["n_val"], "n_test": meta["n_test"],
        "history": history,
        "best_val_acc": best_val_acc,
        "training_time_sec": training_time,
        "param_report": param_report.as_dict(),
    }
    with open(results_dir / "logs" / f"train_{run_name}_seed{seed}.json", "w") as f:
        json.dump(log, f, indent=2)

    return model, log, (train_loader, val_loader, test_loader, classes, meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         choices=["simple_cnn", "resnet18", "mobilenet_v2", "classical_proposed", "hybrid"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--representation", choices=["cnn", "gnn"], default="cnn")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _, log, _ = train_model(args.model, cfg, seed=args.seed, representation=args.representation)
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
