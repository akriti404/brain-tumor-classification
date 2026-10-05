"""
Step 4: Error-targeted Grad-CAM.

Visualizes WHERE in the image the model is looking specifically for
glioma<->meningioma misclassifications -- the confusion pair responsible
for ~85% of all remaining test errors after Step 3 (TTA + ensemble).
Unlike experiments/explainability_cnn.py (which samples the first N test
images regardless of correctness), this script specifically FINDS the
misclassified glioma/meningioma cases and explains those.

Diagnostic purpose: if the heatmaps consistently highlight plausible tumor
regions even on misclassified cases, the confusion is likely a genuinely
hard/ambiguous visual distinction at the single-2D-slice level (a data-
level ceiling, not an architecture problem). If heatmaps are scattered,
off-target, or focus on irrelevant image regions, that points to a model
problem worth more architecture work.

Supports:
  - representation="cnn": classical_proposed or hybrid, via their shared
    `.extractor.features` backbone (LightweightFeatureExtractor).
  - representation="fusion": the fusion_hybrid model, via `.cnn_backbone`.
    Since pytorch_grad_cam's CAM classes call `model(input_tensor)` with a
    single positional argument, and CNNGNNFusionVQC.forward needs
    (images, graph_batch), we wrap the model+fixed-graph per sample in a
    tiny adapter module that exposes a single-argument forward.

NOTE on the existing explainability_cnn.py bug: its get_target_layers()
fallback branch does `for name, module in model.modules()`, but
model.modules() yields modules only, not (name, module) pairs -- that
fallback silently breaks if ever reached. This script's get_target_layers
uses model.named_modules() correctly; worth porting the same fix back into
explainability_cnn.py separately if you keep using it.

Usage:
    python experiments/explainability_errors.py --model classical_proposed --representation cnn --config configs/config.yaml --seed 42
    python experiments/explainability_errors.py --model hybrid --representation cnn --config configs/config.yaml --seed 42
    python experiments/explainability_errors.py --model hybrid --representation fusion --config configs/config.yaml --seed 42

Place this at: experiments/explainability_errors.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, collect_samples, patient_level_split
from models.graph import mri_to_graph
from train import build_model
from utils.reproducibility import set_seed, get_device

TARGET_PAIR_CLASSES = ("glioma", "meningioma")  # the confusion pair we're diagnosing


def get_target_layers(model, representation: str):
    """Find the last Conv2d in the CNN backbone, generically, for either
    representation. Fixes the named_modules() bug present in
    experiments/explainability_cnn.py's fallback branch."""
    if representation == "fusion":
        backbone = model.cnn_backbone
    elif hasattr(model, "extractor"):
        backbone = model.extractor.features
    else:
        raise ValueError(f"Don't know how to find the CNN backbone for representation='{representation}'")

    target_layers = []
    for name, module in reversed(list(backbone.named_modules())):
        if isinstance(module, nn.Conv2d):
            target_layers.append(module)
            print(f"Target layer: {name}")
            break
    if not target_layers:
        raise ValueError("No Conv2d layer found in the backbone")
    return target_layers


class _FusionCamWrapper(nn.Module):
    """Adapts CNNGNNFusionVQC's forward(images, graph_batch) to the single-
    argument forward(images) signature pytorch_grad_cam's CAM classes call."""

    def __init__(self, model, graph_batch):
        super().__init__()
        self.model = model
        self.graph_batch = graph_batch

    def forward(self, images):
        return self.model(images, self.graph_batch)


def denormalize_to_rgb01(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Error-targeted Grad-CAM for glioma<->meningioma confusions")
    parser.add_argument("--model", type=str, required=True, choices=["classical_proposed", "hybrid"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--representation", choices=["cnn", "fusion"], default="cnn")
    parser.add_argument("--max_cases", type=int, default=60,
                         help="Cap on how many misclassified glioma/meningioma cases to render (there are "
                              "~50 in the current error set; this is just a safety cap, not a sampling choice)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(cfg["project"]["device"])

    data_cfg = cfg["data"]
    samples, classes = collect_samples(data_cfg["root"])
    _, _, test_s, _ = patient_level_split(
        samples, data_cfg["val_frac"], data_cfg["test_frac"], seed, data_cfg.get("patient_level_split", True),
    )
    if not all(c in classes for c in TARGET_PAIR_CLASSES):
        raise ValueError(f"Expected classes {TARGET_PAIR_CLASSES} to be present, found {classes}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    target_idxs = {class_to_idx[c] for c in TARGET_PAIR_CLASSES}

    eval_tf = build_transforms(data_cfg["image_size"], data_cfg["augmentation"], train=False)

    n_classes = len(classes)
    model, _ = build_model(args.model, cfg, n_classes, args.representation)
    model.to(device)
    model.eval()

    run_prefix = "" if args.representation == "cnn" else f"{args.representation}_"
    checkpoint_path = Path(cfg["project"]["results_dir"]) / "checkpoints" / f"{run_prefix}{args.model}_seed{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {checkpoint_path}")

    target_layers = get_target_layers(model, args.representation)

    output_dir = Path(args.output_dir or (Path(cfg["project"]["results_dir"]) / "explainability" / "error_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_cfg_full = cfg.get("graph", {})
    graph_cfg = {k: graph_cfg_full[k] for k in ("n_segments", "compactness") if k in graph_cfg_full}

    cases = []
    rendered = 0
    for index, (path, true_label, _pid) in enumerate(test_s):
        if true_label not in target_idxs:
            continue  # only care about the two classes in the confusion pair

        with Image.open(path) as image:
            image = image.convert("RGB")
        image_t = eval_tf(image).to(device)

        graph_batch = None
        if args.representation == "fusion":
            from torch_geometric.data import Batch
            graph = mri_to_graph(image_t.cpu(), seed=seed + index, **graph_cfg).to(device)
            graph_batch = Batch.from_data_list([graph])  # built once, reused for both the
                                                           # prediction check and the CAM pass below

        with torch.no_grad():
            logits = model(image_t.unsqueeze(0), graph_batch) if args.representation == "fusion" \
                else model(image_t.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred_label = probs.argmax(dim=1).item()
            confidence = probs[0, pred_label].item()

        # Only the specific error we're diagnosing: predicted the OTHER class
        # in the target pair, not just "any wrong prediction".
        if pred_label == true_label or pred_label not in target_idxs:
            continue
        if rendered >= args.max_cases:
            continue

        cam_model = _FusionCamWrapper(model, graph_batch) if args.representation == "fusion" else model

        cam = GradCAM(model=cam_model, target_layers=target_layers)
        heatmap = cam(input_tensor=image_t.unsqueeze(0), targets=[ClassifierOutputTarget(pred_label)])[0, :]

        rgb01 = denormalize_to_rgb01(image_t)
        overlay = show_cam_on_image(rgb01, heatmap, use_rgb=True)

        true_name, pred_name = classes[true_label], classes[pred_label]
        out_path = output_dir / f"err_{index:04d}_true-{true_name}_pred-{pred_name}_conf{confidence:.2f}.png"
        Image.fromarray(overlay).save(out_path)

        cases.append({
            "test_index": index, "source_path": str(path), "true_class": true_name,
            "pred_class": pred_name, "confidence": confidence, "heatmap_path": str(out_path),
        })
        rendered += 1
        print(f"  [{rendered}] true={true_name} pred={pred_name} conf={confidence:.3f} -> {out_path.name}")

    summary_path = output_dir / f"error_analysis_{run_prefix}{args.model}_seed{seed}.json"
    with open(summary_path, "w") as f:
        json.dump({"n_cases": len(cases), "cases": cases}, f, indent=2)

    print(f"\nFound and rendered {len(cases)} glioma<->meningioma misclassifications.")
    print(f"Heatmaps: {output_dir}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()