"""
Trains one model (a classical baseline or the proposed hybrid CNN+VQC) as
specified by --model, using the shared data pipeline and config.
  - New --representation choice: "fusion" (in addition to "cnn", "gnn").
    Only valid with --model hybrid. Uses models.fusion.CNNGNNFusionVQC and
    data.fusion_dataset.build_fusion_dataloaders, whose batches are
    (images, graph_batch, labels) tuples rather than a single object.
  - run_one_epoch now dispatches explicitly by `representation` instead of
    duck-typing on batch shape (hasattr(batch, "edge_index")) -- the old
    duck-typing approach can't distinguish "gnn" from "fusion" batches.
  - Checkpoint/log naming generalized: previously only "gnn" got a
    representation prefix; now anything other than "cnn" does
    (run_name = model_name if representation == "cnn" else
    f"{representation}_{model_name}"), so fusion checkpoints save as
    fusion_hybrid_seed<seed>.pt without colliding with hybrid_seed<seed>.pt
    or gnn_hybrid_seed<seed>.pt.
  - Fusion's CNN backbone also gets the Step 1 two-stage freeze/unfreeze
    schedule (it has a pretrained MobileNetV2 backbone same as the CNN
    branch); the GNN/cross-attention/fusion/VQC parts train from scratch
    throughout, same as the existing GNN branch.

Usage:
    python train.py --model hybrid --representation cnn --config configs/config.yaml
    python train.py --model hybrid --representation gnn --config configs/config.yaml
    python train.py --model hybrid --representation fusion --config configs/config.yaml
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
from data.fusion_dataset import build_fusion_dataloaders
from models.classical import build_baseline, ClassicalProposedHead
from models.graph import build_graph_dataloaders
from models.hybrid import build_model_from_config
from models.gnn_hybrid import build_gnn_model_from_config
from models.fusion import build_fusion_model_from_config
from utils.reproducibility import set_seed, get_device
from utils.param_count import build_param_report


def build_model(model_name: str, cfg: dict, n_classes: int, representation: str = "cnn"):
    if representation == "gnn":
        if model_name != "hybrid":
            raise ValueError("The gnn representation is only available for --model hybrid")
        return build_gnn_model_from_config(cfg, n_classes), True
    if representation == "fusion":
        if model_name != "hybrid":
            raise ValueError("The fusion representation is only available for --model hybrid")
        return build_fusion_model_from_config(cfg, n_classes), True
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


def get_backbone_and_head_params(model, model_name: str, representation: str):
    """
    Returns (backbone_params, head_params, backbone_module) for models with a
    pretrained backbone that benefits from a two-stage freeze/unfreeze
    schedule, or None if the model has no such backbone.
    """
    if representation == "gnn":
        return None  # GCN is trained from scratch, no pretrained backbone to freeze

    if representation == "fusion":
        # CNNGNNFusionVQC.cnn_backbone is the pretrained MobileNetV2.features;
        # everything else (GNN encoder, cross-attention, fusion gate, reducer,
        # qlayer, classifier) trains from scratch throughout.
        backbone_params = list(model.cnn_backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        return backbone_params, head_params, model.cnn_backbone

    if model_name in ("hybrid", "classical_proposed"):
        backbone_module = model.extractor.features
        backbone_params = list(backbone_module.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        return backbone_params, head_params, backbone_module

    if model_name == "resnet18":
        backbone_module = nn.Sequential(*[m for name, m in model.named_children() if name != "fc"])
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
        head_params = [p for name, p in model.named_parameters() if name.startswith("fc.")]
        return backbone_params, head_params, backbone_module

    if model_name == "mobilenet_v2":
        backbone_module = model.features
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
        return backbone_params, head_params, backbone_module

    return None  # simple_cnn: trained from scratch, no freeze/unfreeze needed


def set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad = flag


def _forward_batch(model, batch, device, representation: str):
    """Returns (logits, y) for one batch, dispatched explicitly by representation
    rather than duck-typing on batch shape."""
    if representation == "fusion":
        images, graph_batch, y = batch
        images, graph_batch, y = images.to(device), graph_batch.to(device), y.to(device)
        return model(images, graph_batch), y
    if representation == "gnn":
        inputs = batch.to(device)
        return model(inputs), inputs.y
    inputs, y = batch[0].to(device), batch[1].to(device)
    return model(inputs), y


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool, representation: str = "cnn"):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if train:
                optimizer.zero_grad()
            logits, y = _forward_batch(model, batch, device, representation)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def update_early_stopping(best_val_acc: float, current_val_acc: float, epochs_without_improvement: int,
                         patience: int, min_delta: float = 1e-4):
    if current_val_acc > best_val_acc + min_delta:
        return current_val_acc, 0, False
    return best_val_acc, epochs_without_improvement + 1, patience > 0 and (epochs_without_improvement + 1) >= patience


def _run_phase(model, model_name, representation, train_loader, val_loader, criterion, optimizer, scheduler,
               device, n_epochs, history, best_val_acc, best_state, epochs_without_improvement, patience,
               epoch_offset, total_epochs):
    """Runs `n_epochs` of train/val, mutating and returning the shared bookkeeping state."""
    stopped_early = False
    for local_epoch in range(n_epochs):
        global_epoch = epoch_offset + local_epoch
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, True, representation)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, False, representation)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{model_name}/{representation}] epoch {global_epoch + 1}/{total_epochs} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"[Early stopping] no validation improvement for {patience} epochs at epoch "
                  f"{global_epoch + 1}; stopping.")
            stopped_early = True
            break

    return best_val_acc, best_state, epochs_without_improvement, stopped_early


def train_model(model_name: str, cfg: dict, seed: int = None, representation: str = "cnn"):
    seed = seed if seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(cfg["project"]["device"])

    if representation == "gnn":
        loaders = build_graph_dataloaders(cfg)
    elif representation == "fusion":
        loaders = build_fusion_dataloaders(cfg)
    else:
        loaders = build_dataloaders(cfg, seed=seed)
    train_loader, val_loader, test_loader, classes, meta = loaders
    n_classes = len(classes)

    model, is_quantum = build_model(model_name, cfg, n_classes, representation)
    model.to(device)

    class_weights = meta["class_weights"]
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=label_smoothing,
    )

    total_epochs = cfg["training"]["epochs"]
    warmup_epochs = int(cfg["training"].get("warmup_epochs", 0))
    backbone_split = get_backbone_and_head_params(model, model_name, representation)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    patience = int(cfg["training"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    t0 = time.time()

    if backbone_split is not None and warmup_epochs > 0:
        backbone_params, head_params, _ = backbone_split
        head_lr = float(cfg["training"].get("head_lr", cfg["training"]["lr"]))
        backbone_lr = float(cfg["training"].get("backbone_lr", cfg["training"]["lr"] * 0.1))

        # --- Phase A: backbone frozen, head-only warmup at head_lr ---
        set_requires_grad(backbone_params, False)
        warmup_optimizer = torch.optim.Adam(head_params, lr=head_lr, weight_decay=cfg["training"]["weight_decay"])
        best_val_acc, best_state, epochs_without_improvement, stopped = _run_phase(
            model, model_name, representation, train_loader, val_loader, criterion, warmup_optimizer,
            scheduler=None, device=device, n_epochs=min(warmup_epochs, total_epochs), history=history,
            best_val_acc=best_val_acc, best_state=best_state,
            epochs_without_improvement=epochs_without_improvement, patience=patience,
            epoch_offset=0, total_epochs=total_epochs,
        )

        remaining_epochs = total_epochs - min(warmup_epochs, total_epochs)
        if not stopped and remaining_epochs > 0:
            # --- Phase B: unfreeze backbone, differential LR, cosine over the remainder ---
            set_requires_grad(backbone_params, True)
            optimizer = torch.optim.Adam(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": head_params, "lr": head_lr},
                ],
                weight_decay=cfg["training"]["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            best_val_acc, best_state, epochs_without_improvement, _ = _run_phase(
                model, model_name, representation, train_loader, val_loader, criterion, optimizer, scheduler,
                device=device, n_epochs=remaining_epochs, history=history,
                best_val_acc=best_val_acc, best_state=best_state,
                epochs_without_improvement=epochs_without_improvement, patience=patience,
                epoch_offset=min(warmup_epochs, total_epochs), total_epochs=total_epochs,
            )
    else:
        # Single-phase training (no pretrained backbone to warm up, or warmup disabled).
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        best_val_acc, best_state, epochs_without_improvement, _ = _run_phase(
            model, model_name, representation, train_loader, val_loader, criterion, optimizer, scheduler,
            device=device, n_epochs=total_epochs, history=history,
            best_val_acc=best_val_acc, best_state=best_state,
            epochs_without_improvement=epochs_without_improvement, patience=patience,
            epoch_offset=0, total_epochs=total_epochs,
        )

    training_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # Generalized naming: only "cnn" (the original default) has no prefix,
    # so existing hybrid_seed42.pt / train_hybrid_seed42.json paths from
    # Step 1 are unaffected; "gnn" and "fusion" each get their own prefix.
    run_name = model_name if representation == "cnn" else f"{representation}_{model_name}"

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
        "warmup_epochs_used": min(warmup_epochs, total_epochs) if backbone_split is not None else 0,
        "label_smoothing": label_smoothing,
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
    parser.add_argument("--representation", choices=["cnn", "gnn", "fusion"], default="cnn")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _, log, _ = train_model(args.model, cfg, seed=args.seed, representation=args.representation)
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
