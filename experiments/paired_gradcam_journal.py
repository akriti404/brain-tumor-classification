"""Create a paired Grad-CAM figure on a shared, audited test cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from train import build_model
from utils.reproducibility import get_device, set_seed


CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
MODEL_NAMES = {
    "classical_proposed": "Classical",
    "hybrid": "Hybrid CNN-VQC",
}
def _relative_key(path: str) -> str:
    parts = Path(path.replace("\\", "/")).parts
    for index, part in enumerate(parts):
        if part.lower() in CLASSES:
            return Path(*parts[index:]).as_posix().lower()
    raise ValueError(f"Prediction path does not contain a recognized class: {path}")


def _load_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return {_relative_key(row["image_path"]): row for row in csv.DictReader(file)}


def _load_audited_cases(
    audit_path: Path,
    prediction_dir: Path,
    test_root: Path,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, dict]]]:
    if not audit_path.exists():
        raise FileNotFoundError(f"Test integrity audit not found: {audit_path}")
    with audit_path.open(newline="", encoding="utf-8") as file:
        retained = {
            row["image_path"].replace("/", "\\").lower(): row
            for row in csv.DictReader(file)
            if row["status"] == "retained_unique"
        }
    if not retained:
        raise ValueError(f"No retained_unique test files in {audit_path}")

    predictions = {
        model: _load_csv(prediction_dir / f"{model}_seed{seed}.csv")
        for model in MODEL_NAMES
    }
    cases = []
    for relative, audit_row in sorted(retained.items()):
        key = relative.replace("\\", "/").lower()
        model_rows = {}
        for model in MODEL_NAMES:
            if key not in predictions[model]:
                raise ValueError(f"{model} predictions are missing audited test image: {relative}")
            model_rows[model] = predictions[model][key]

        true_class = relative.split("\\", 1)[0]
        if true_class not in CLASSES or any(
            row["true_class"] != true_class for row in model_rows.values()
        ):
            raise ValueError(f"Class label mismatch for audited test image: {relative}")
        image_path = test_root.joinpath(*relative.split("\\"))
        if not image_path.is_file():
            raise FileNotFoundError(f"Audited test image is missing: {image_path}")
        cases.append({
            "relative_path": relative,
            "image_path": image_path,
            "true_class": true_class,
            "predictions": model_rows,
            "audit_sha256": audit_row["sha256"],
        })

    if len(cases) != len(retained):
        raise ValueError("Not every retained audit entry was loaded.")
    return cases, predictions


def _select_cases(cases: list[dict], max_cases_per_class: int = 4) -> dict[str, list[dict]]:
    if max_cases_per_class < 1:
        raise ValueError("At least one case per class is required.")
    selected = []
    for true_class in CLASSES:
        class_cases = [case for case in cases if case["true_class"] == true_class]
        both_correct = [
            case for case in class_cases
            if all(
                row["predicted_class"] == true_class
                for row in case["predictions"].values()
            )
        ]
        both_correct.sort(key=lambda case: (
            -float(case["predictions"]["hybrid"]["probability_" + true_class]),
            case["relative_path"],
        ))

        if not both_correct:
            raise ValueError(
                f"No image correctly classified by both models for {true_class}."
            )
        class_selected = [
            {**case, "example_type": "both_models_correct_high_hybrid_confidence"}
            for case in both_correct[:max_cases_per_class]
        ]
        selected.append((true_class, class_selected))
    return dict(selected)


def _target_layer(model: nn.Module) -> nn.Module:
    backbone = model.extractor.features
    if len(backbone) <= 13:
        raise ValueError("MobileNetV2 backbone does not expose the expected Grad-CAM layer.")
    return backbone[13]


def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype).view(3, 1, 1)
    image = (image_tensor.detach().cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def _prediction(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        return predicted_index, float(probabilities[predicted_index].item())


def _gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    image_tensor: torch.Tensor,
    class_index: int,
    device: torch.device,
) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=[target_layer])
    values = cam(
        input_tensor=image_tensor.unsqueeze(0).to(device),
        targets=[ClassifierOutputTarget(class_index)],
    )
    return values[0]


def _write_case_manifest(path: Path, selected: list[dict], class_names: list[str]) -> None:
    rows = []
    for case in selected:
        row = {
            "example_type": case["example_type"],
            "image_path": case["relative_path"],
            "true_class": case["true_class"],
            "audit_sha256": case["audit_sha256"],
        }
        for model, label in MODEL_NAMES.items():
            prediction = case["predictions"][model]
            row[f"{model}_predicted_class"] = prediction["predicted_class"]
            row[f"{model}_confidence"] = prediction[
                f"probability_{prediction['predicted_class']}"
            ]
            row[f"{model}_correct"] = prediction["predicted_class"] == case["true_class"]
        rows.append(row)

    if class_names != list(CLASSES):
        raise ValueError(f"Unexpected class ordering in config: {class_names}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(args: argparse.Namespace) -> None:
    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    seed = args.seed
    set_seed(seed)
    device = get_device(cfg["project"]["device"])
    classes = sorted(CLASSES)

    audited_cases, _ = _load_audited_cases(
        args.audit_csv, args.prediction_dir, args.test_root, seed
    )
    selected_by_class = _select_cases(audited_cases)

    models = {}
    target_layers = {}
    for model_name in MODEL_NAMES:
        model, _ = build_model(model_name, cfg, len(classes), representation="cnn")
        checkpoint = args.checkpoint_dir / f"{model_name}_seed{seed}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.to(device).eval()
        models[model_name] = model
        target_layers[model_name] = _target_layer(model)

    eval_transform = build_transforms(
        cfg["data"]["image_size"],
        cfg["data"]["augmentation"],
        train=False,
        crop_margin=cfg["data"].get("margin_crop", False),
    )
    class_to_index = {name: index for index, name in enumerate(classes)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = []
    selected = [case for cases_for_class in selected_by_class.values() for case in cases_for_class]
    for true_class, class_cases in selected_by_class.items():
        fig, axes = plt.subplots(
            len(class_cases), 3,
            figsize=(10.5, 2.25 * len(class_cases)),
            squeeze=False,
            constrained_layout=True,
        )
        for column, title in enumerate(
            ("Original MRI", "Classical Grad-CAM", "Hybrid CNN-VQC Grad-CAM")
        ):
            axes[0, column].set_title(title, fontsize=10, weight="bold")

        for row_index, case in enumerate(class_cases):
            image_digest = hashlib.sha256(case["image_path"].read_bytes()).hexdigest().upper()
            if image_digest != case["audit_sha256"].upper():
                raise ValueError(
                    f"Audited test image has changed since duplicate checking: {case['image_path']}"
                )
            with Image.open(case["image_path"]) as source:
                original = source.convert("RGB")
            image_tensor = eval_transform(original)
            rgb01 = _denormalize(image_tensor)
            axes[row_index, 0].imshow(original, cmap="gray")
            axes[row_index, 0].set_ylabel(
                f"Both models correct\n{case['image_path'].name}",
                fontsize=8,
            )

            actual_outputs = {}
            for model_name, column in (("classical_proposed", 1), ("hybrid", 2)):
                model = models[model_name]
                predicted_index, confidence = _prediction(model, image_tensor, device)
                predicted_class = classes[predicted_index]
                recorded = case["predictions"][model_name]
                if predicted_class != recorded["predicted_class"]:
                    raise ValueError(
                        f"Checkpoint prediction does not match saved evaluation for "
                        f"{case['relative_path']} ({model_name}): {predicted_class} != "
                        f"{recorded['predicted_class']}"
                    )
                expected_confidence = float(recorded[f"probability_{predicted_class}"])
                if not np.isclose(confidence, expected_confidence, rtol=1e-3, atol=1e-3):
                    raise ValueError(
                        f"Checkpoint confidence does not match saved evaluation for "
                        f"{case['relative_path']} ({model_name}): {confidence:.5f} != "
                        f"{expected_confidence:.5f}"
                    )

                heatmap = _gradcam(
                    model, target_layers[model_name], image_tensor, predicted_index, device
                )
                overlay = show_cam_on_image(rgb01, heatmap, use_rgb=True)
                axes[row_index, column].imshow(overlay)
                axes[row_index, column].set_title(
                    f"True: {true_class.title()} | Pred: {predicted_class.title()}\n"
                    f"Confidence: {confidence:.1%}",
                    fontsize=8,
                )
                actual_outputs[model_name] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "correct": predicted_class == case["true_class"],
                }

            for column in range(3):
                axes[row_index, column].axis("off")
            case_manifest.append({
                "example_type": case["example_type"],
                "image_path": case["relative_path"],
                "sha256": case["audit_sha256"],
                "true_class": case["true_class"],
                "predictions": actual_outputs,
            })

        fig.suptitle(
            f"Paired Grad-CAM on audited official test images: {true_class.title()}",
            fontsize=12,
        )
        figure_path = args.output_dir / f"paired_gradcam_{true_class}_seed{seed}"
        fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved class Grad-CAM figure: {figure_path.with_suffix('.png')}")

    _write_case_manifest(args.output_dir / "selected_cases.csv", selected, classes)
    with (args.output_dir / "selected_cases.json").open("w", encoding="utf-8") as file:
        json.dump({
            "seed": seed,
            "test_root": str(args.test_root),
            "audit_csv": str(args.audit_csv),
            "selection_rule": (
                "For each true class, select up to four images correctly classified by both "
                "models, ranked by descending hybrid true-class confidence."
            ),
            "interpretation_limit": (
                "Only high-confidence cases correctly classified by both models are shown. "
                "The dataset has no tumor masks or bounding-box annotations, so Grad-CAM "
                "focus is qualitative and does not establish tumor-localization accuracy."
            ),
            "preprocessing": {
                "image_size": cfg["data"]["image_size"],
                "margin_crop": cfg["data"].get("margin_crop", False),
                "normalization": "ImageNet mean and standard deviation",
                "gradcam_target_layer": "MobileNetV2.features[13]",
            },
            "cases": case_manifest,
        }, file, indent=2)
    print(f"Saved selected-case manifest: {args.output_dir / 'selected_cases.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a same-image classical vs hybrid CNN-VQC Grad-CAM figure."
    )
    parser.add_argument("--config", type=Path, default=Path("results/epochs50/run_config.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_root", type=Path,
        default=Path("data/kaggle_test_unique_train_disjoint"),
    )
    parser.add_argument(
        "--audit_csv", type=Path,
        default=Path("results/tables/official_test_integrity_audit.csv"),
    )
    parser.add_argument(
        "--prediction_dir", type=Path,
        default=Path("results/epochs50/predictions/unique_test"),
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path,
        default=Path("results/epochs50/checkpoints"),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("results/explainability/journal_paired_seed42"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
