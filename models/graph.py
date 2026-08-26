"""SLIC-based MRI graph construction and graph data loading."""
from pathlib import Path
import inspect

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    collect_samples,
    compute_class_weights,
    make_weighted_sampler,
    patient_level_split,
)


def _to_display_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert one ImageNet-normalized CHW tensor to an RGB float image."""
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).cpu().numpy()


def _slic_segments(image: np.ndarray, n_segments: int, compactness: float, seed: int) -> np.ndarray:
    """Call SLIC with deterministic randomness across supported scikit-image versions."""
    slic_params = inspect.signature(slic).parameters
    if "rng" in slic_params:
        return slic(image, n_segments=n_segments, compactness=compactness,
                    channel_axis=-1, start_label=0, rng=np.random.default_rng(seed))
    if "random_seed" in slic_params:
        return slic(image, n_segments=n_segments, compactness=compactness,
                    channel_axis=-1, start_label=0, random_seed=seed)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return slic(image, n_segments=n_segments, compactness=compactness,
                    channel_axis=-1, start_label=0)
    finally:
        np.random.set_state(state)


def mri_to_graph(image: torch.Tensor, n_segments: int = 32, compactness: float = 10.0,
                 seed: int = 42) -> Data:
    """Convert one preprocessed MRI tensor into a standardized PyG graph.

    Node features are mean RGB, grayscale standard deviation, normalized centroid,
    and area fraction. Edges connect superpixels that touch in the image plane.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW tensor, got shape {tuple(image.shape)}")

    rgb = _to_display_rgb(image)
    segments = _slic_segments(rgb, n_segments, compactness, seed)
    grayscale = rgb.mean(axis=2)
    labels = np.unique(segments)
    label_to_node = {int(label): i for i, label in enumerate(labels)}
    features = []

    for label in labels:
        mask = segments == label
        rows, cols = np.where(mask)
        pixels = rgb[mask]
        features.append([
            *pixels.mean(axis=0),
            float(grayscale[mask].std()),
            float(rows.mean() / max(rgb.shape[0] - 1, 1)),
            float(cols.mean() / max(rgb.shape[1] - 1, 1)),
            float(mask.mean()),
        ])

    node_features = torch.tensor(features, dtype=torch.float32)
    node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    adjacent = set()
    for axis in (0, 1):
        left = np.take(segments, indices=range(segments.shape[axis] - 1), axis=axis)
        right = np.take(segments, indices=range(1, segments.shape[axis]), axis=axis)
        for source, target in zip(left.ravel(), right.ravel()):
            source, target = int(source), int(target)
            if source != target:
                adjacent.add(tuple(sorted((label_to_node[source], label_to_node[target]))))

    if adjacent:
        edge_pairs = list(adjacent)
        edge_index = torch.tensor(edge_pairs + [(b, a) for a, b in edge_pairs], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index)


class GraphMRIDataset(Dataset):
    def __init__(self, samples, transform, graph_cfg, seed):
        self.samples = samples
        self.transform = transform
        self.graph_cfg = graph_cfg
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = self.transform(image)
        graph = mri_to_graph(image, seed=self.seed + index, **self.graph_cfg)
        graph.y = torch.tensor(label, dtype=torch.long)
        return graph


def build_graph_dataloaders(cfg: dict):
    """Build graph loaders using the same samples, splits, transforms, and seed."""
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
        GraphMRIDataset(train_s, train_tf, slic_cfg, cfg["project"]["seed"]),
        GraphMRIDataset(val_s, eval_tf, slic_cfg, cfg["project"]["seed"]),
        GraphMRIDataset(test_s, eval_tf, slic_cfg, cfg["project"]["seed"]),
    ]
    strategy = data_cfg.get("class_imbalance_strategy", "none")
    sampler = make_weighted_sampler(train_s) if strategy == "weighted_sampler" else None
    class_weights = compute_class_weights(train_s, len(classes)) if strategy == "class_weighted_loss" else None
    common = {"batch_size": data_cfg["batch_size"], "num_workers": data_cfg["num_workers"]}
    loaders = [
        DataLoader(datasets[0], sampler=sampler, shuffle=sampler is None, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    ]
    meta = {"classes": classes, "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
            "split_method": split_method, "class_weights": class_weights}
    return *loaders, classes, meta