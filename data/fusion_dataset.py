"""
Dataset/dataloader for the CNN+GNN fusion branch (Step 2).

Unlike data/dataset.py (image-only) or models/graph.py (graph-only), the
fusion model needs BOTH the raw image tensor (for the CNN spatial feature
map) and the SLIC graph (for the GNN) for the *same* sample in the same
forward pass, so cross-attention can align them. This module reuses the
existing sample discovery / patient-level split / transforms / class-weight
utilities from data/dataset.py and the graph construction from
models/graph.py — it does not reimplement any of that, only combines them.

Place this at: data/fusion_dataset.py
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from data.dataset import (
    build_transforms,
    collect_samples,
    compute_class_weights,
    make_weighted_sampler,
    patient_level_split,
)
from models.graph import mri_to_graph


class FusionMRIDataset(Dataset):
    """Returns (image_tensor, graph_data, label) per sample."""

    def __init__(self, samples, transform, graph_cfg, seed):
        self.samples = samples
        self.transform = transform
        self.graph_cfg = graph_cfg
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _pid = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        image_t = self.transform(image)
        graph = mri_to_graph(image_t, seed=self.seed + index, **self.graph_cfg)
        return image_t, graph, label


def fusion_collate(batch):
    """
    Custom collate: images stack normally, graphs batch via PyG's Batch
    (which offsets node indices and builds the `.batch` assignment vector
    the fusion model needs to know which nodes belong to which image),
    labels stack as a plain LongTensor.
    """
    images, graphs, labels = zip(*batch)
    image_batch = torch.stack(images, dim=0)
    graph_batch = Batch.from_data_list(list(graphs))
    label_batch = torch.tensor(labels, dtype=torch.long)
    return image_batch, graph_batch, label_batch


def build_fusion_dataloaders(cfg: dict):
    """Mirrors models.graph.build_graph_dataloaders but yields (image, graph, label) batches."""
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    if not root.exists() or not any(root.iterdir()):
        if data_cfg.get("synthetic_fallback", False):
            from data.synthetic_data_generator import generate_synthetic_dataset
            generate_synthetic_dataset(str(root), image_size=data_cfg["image_size"])
        else:
            raise FileNotFoundError(f"Dataset root '{root}' not found and synthetic_fallback is disabled.")

    samples, classes = collect_samples(str(root))
    train_s, val_s, test_s, split_method = patient_level_split(
        samples, data_cfg["val_frac"], data_cfg["test_frac"],
        cfg["project"]["seed"], data_cfg.get("patient_level_split", True),
    )
    train_tf = build_transforms(data_cfg["image_size"], data_cfg["augmentation"], train=True)
    eval_tf = build_transforms(data_cfg["image_size"], data_cfg["augmentation"], train=False)
    graph_cfg = cfg.get("graph", {})
    slic_cfg = {key: graph_cfg[key] for key in ("n_segments", "compactness") if key in graph_cfg}

    datasets = [
        FusionMRIDataset(train_s, train_tf, slic_cfg, cfg["project"]["seed"]),
        FusionMRIDataset(val_s, eval_tf, slic_cfg, cfg["project"]["seed"]),
        FusionMRIDataset(test_s, eval_tf, slic_cfg, cfg["project"]["seed"]),
    ]
    strategy = data_cfg.get("class_imbalance_strategy", "none")
    sampler = make_weighted_sampler(train_s) if strategy == "weighted_sampler" else None
    class_weights = compute_class_weights(train_s, len(classes)) if strategy == "class_weighted_loss" else None
    common = {"batch_size": data_cfg["batch_size"], "num_workers": data_cfg["num_workers"],
              "collate_fn": fusion_collate}
    loaders = [
        DataLoader(datasets[0], sampler=sampler, shuffle=sampler is None, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    ]
    meta = {"classes": classes, "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
            "split_method": split_method, "class_weights": class_weights}
    return *loaders, classes, meta
