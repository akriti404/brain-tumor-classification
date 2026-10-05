"""
Dataset pipeline for the ImageFolder-style MRI dataset.

Responsibilities (spec Section 3):
  - auto-discover classes (never assumed)
  - patient-level split when patient IDs are inferable from filenames,
    falling back to stratified image-level split otherwise, to avoid
    leakage from multiple slices of the same patient crossing splits
  - resizing / normalization / augmentation
  - class-imbalance handling via a WeightedRandomSampler (or class-weighted
    loss, configurable)

Step 4 addition: CropToContent, a margin-removal preprocessing transform.

The Kaggle Brain Tumor MRI Dataset (masoudnickparvar/brain-tumor-mri-dataset)
is a merge of three source datasets with different native resolutions and
framing conventions (glioma from figshare; meningioma/pituitary from SARTAJ;
notumor from Br35H -- see data/dataset_inspector.py output: image sizes
span 512x512, 225x225, 96x96, 630x630, 236x236, and more). The dataset's own
documentation explicitly recommends "removing the extra margins" before
training, because without it a model can learn to key off border/framing
artifacts specific to each source rather than actual tumor morphology --
exactly what error-targeted Grad-CAM analysis (experiments/
explainability_errors.py) showed was happening on glioma<->meningioma
misclassifications: heatmaps landing on skull edges and image corners
rather than brain tissue.

CropToContent crops away the near-black margin around the head/brain
BEFORE resizing, using an Otsu threshold on grayscale intensity to find
the foreground bounding box. It is inserted as the first step of
build_transforms and is therefore applied identically across the CNN,
GNN, and fusion data pipelines (all three call build_transforms from this
module) -- a single change point rather than three separate ones.
"""
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PATIENT_ID_PATTERN = re.compile(r"(patient[_\-]?\d+|pid[_\-]?\d+|p\d{3,})", re.IGNORECASE)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropToContent:
    """
    Crops away black/near-black margin surrounding the head in brain MRI
    images, before resizing. Falls back to returning the image unmodified
    if Otsu thresholding fails to find a sensible foreground region (e.g.
    a near-uniform image) -- preprocessing on a handful of edge-case
    images should degrade gracefully, not crash a training run.
    """

    def __init__(self, padding_frac: float = 0.02, min_content_frac: float = 0.05):
        self.padding_frac = padding_frac
        self.min_content_frac = min_content_frac

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        if gray.max() <= gray.min():  # degenerate (uniform) image, nothing to crop
            return image
        try:
            thresh = threshold_otsu(gray)
        except ValueError:
            return image

        mask = gray > thresh
        foreground_frac = mask.mean()
        if foreground_frac < self.min_content_frac or foreground_frac > (1 - self.min_content_frac):
            # Foreground is implausibly small or implausibly large -- thresholding
            # likely failed on this particular image. Skip cropping rather than
            # risk cropping away actual anatomy.
            return image

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_idxs = np.where(rows)[0]
        col_idxs = np.where(cols)[0]
        if row_idxs.size == 0 or col_idxs.size == 0:
            return image

        top, bottom = int(row_idxs[0]), int(row_idxs[-1])
        left, right = int(col_idxs[0]), int(col_idxs[-1])

        h, w = gray.shape
        pad_h = int(h * self.padding_frac)
        pad_w = int(w * self.padding_frac)
        top = max(0, top - pad_h)
        bottom = min(h - 1, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(w - 1, right + pad_w)

        return image.crop((left, top, right + 1, bottom + 1))


def discover_classes(root: str):
    root_path = Path(root)
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    return [d.name for d in class_dirs]


def _infer_patient_id(filename: str) -> str:
    m = PATIENT_ID_PATTERN.search(filename)
    return m.group(1).lower() if m else filename  # fallback: treat each file as its own "patient"


def collect_samples(root: str):
    """Returns list of (path, class_idx, patient_id) and the class list."""
    classes = discover_classes(root)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for cls in classes:
        cls_dir = Path(root) / cls
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in IMG_EXTENSIONS:
                pid = _infer_patient_id(f.stem)
                samples.append((str(f), class_to_idx[cls], pid))
    return samples, classes


def patient_level_split(samples, val_frac: float, test_frac: float, seed: int, use_patient_level: bool):
    """
    Splits samples into train/val/test. If patient IDs are inferable and
    use_patient_level=True, splitting happens at the patient level so all
    slices from one patient stay in one split (avoids leakage). Falls back
    to stratified image-level split otherwise.
    """
    unique_pids = {s[2] for s in samples}
    has_real_patient_structure = use_patient_level and len(unique_pids) < len(samples)

    if has_real_patient_structure:
        # Assign each patient a majority-class label for stratification, then split patients.
        pid_to_labels = defaultdict(list)
        for _, label, pid in samples:
            pid_to_labels[pid].append(label)
        pids = list(pid_to_labels.keys())
        pid_labels = [max(set(v), key=v.count) for v in pid_to_labels.values()]

        train_pids, temp_pids, train_lbl, temp_lbl = train_test_split(
            pids, pid_labels, test_size=(val_frac + test_frac), random_state=seed, stratify=pid_labels
        )
        rel_test = test_frac / (val_frac + test_frac)
        val_pids, test_pids = train_test_split(
            temp_pids, test_size=rel_test, random_state=seed, stratify=temp_lbl
        )
        train_pids, val_pids, test_pids = set(train_pids), set(val_pids), set(test_pids)

        train = [s for s in samples if s[2] in train_pids]
        val = [s for s in samples if s[2] in val_pids]
        test = [s for s in samples if s[2] in test_pids]
        split_method = "patient_level"
    else:
        labels = [s[1] for s in samples]
        train, temp, train_lbl, temp_lbl = train_test_split(
            samples, labels, test_size=(val_frac + test_frac), random_state=seed, stratify=labels
        )
        rel_test = test_frac / (val_frac + test_frac)
        val, test = train_test_split(temp, test_size=rel_test, random_state=seed, stratify=temp_lbl)
        split_method = "stratified_image_level"

    return train, val, test, split_method


class MRIDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _pid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(image_size: int, augmentation_cfg: dict, train: bool, crop_margin: bool = True):
    ops = []
    if crop_margin:
        ops.append(CropToContent())
    ops.append(transforms.Resize((image_size, image_size)))
    if train:
        if augmentation_cfg.get("horizontal_flip", False):
            ops.append(transforms.RandomHorizontalFlip())
        deg = augmentation_cfg.get("rotation_degrees", 0)
        if deg:
            ops.append(transforms.RandomRotation(deg))
        jitter = augmentation_cfg.get("brightness_jitter", 0)
        if jitter:
            ops.append(transforms.ColorJitter(brightness=jitter))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def make_weighted_sampler(samples) -> WeightedRandomSampler:
    labels = np.array([s[1] for s in samples])
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_class_weights(samples, n_classes) -> torch.Tensor:
    labels = np.array([s[1] for s in samples])
    counts = np.bincount(labels, minlength=n_classes)
    weights = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg: dict, seed: int = None):
    """
    Top-level entry point. Reads cfg['data'], discovers/splits the dataset,
    and returns (train_loader, val_loader, test_loader, classes, meta).
    """
    data_cfg = cfg["data"]
    root = data_cfg["root"]

    root_is_empty = (not Path(root).exists()) or (not any(Path(root).iterdir()))
    if root_is_empty:
        if data_cfg.get("synthetic_fallback", False):
            from data.synthetic_data_generator import generate_synthetic_dataset
            generate_synthetic_dataset(root, image_size=data_cfg["image_size"])
        else:
            raise FileNotFoundError(f"Dataset root '{root}' not found and synthetic_fallback is disabled.")

    samples, classes = collect_samples(root)
    split_seed = cfg["project"]["seed"] if seed is None else seed
    train_s, val_s, test_s, split_method = patient_level_split(
        samples, data_cfg["val_frac"], data_cfg["test_frac"],
        split_seed, data_cfg.get("patient_level_split", True)
    )

    crop_margin = data_cfg.get("margin_crop", True)
    train_tf = build_transforms(data_cfg["image_size"], data_cfg["augmentation"], train=True, crop_margin=crop_margin)
    eval_tf = build_transforms(data_cfg["image_size"], data_cfg["augmentation"], train=False, crop_margin=crop_margin)

    train_ds = MRIDataset(train_s, transform=train_tf)
    val_ds = MRIDataset(val_s, transform=eval_tf)
    test_ds = MRIDataset(test_s, transform=eval_tf)

    imbalance_strategy = data_cfg.get("class_imbalance_strategy", "none")
    sampler = make_weighted_sampler(train_s) if imbalance_strategy == "weighted_sampler" else None
    class_weights = (
        compute_class_weights(train_s, len(classes)) if imbalance_strategy == "class_weighted_loss" else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"], sampler=sampler,
        shuffle=(sampler is None), num_workers=data_cfg["num_workers"]
    )
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])

    meta = {
        "classes": classes,
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
        "split_method": split_method,
        "class_weights": class_weights,
    }
    return train_loader, val_loader, test_loader, classes, meta