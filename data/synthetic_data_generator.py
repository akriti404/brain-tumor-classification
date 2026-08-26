"""
SYNTHETIC PLACEHOLDER DATA GENERATOR — NOT REAL MRI DATA.

This sandbox has no network access to Kaggle / Mendeley / figshare, where the
real public brain-tumor MRI datasets are hosted (network here is restricted to
package registries and github.com code, not data-hosting sites).

This module exists ONLY so the rest of the pipeline (data loading, splitting,
augmentation, baselines, hybrid model, training loop, metrics) can be executed
and verified end-to-end in this environment. It procedurally generates small
grayscale images with class-conditional geometric "lesion" patterns (blobs of
varying position/size/texture per class) — enough structure for a model to
learn something non-trivial, so a smoke test is meaningful, but this is NOT
clinical data and results from it must never be reported as real findings.

To run on the real dataset: download the Kaggle "Brain Tumor MRI Dataset"
(masoudnickparvar/brain-tumor-mri-dataset) or an equivalent public dataset,
arrange it as data/raw/<class_name>/*.jpg, and set data.root in config.yaml.
Nothing else in the pipeline needs to change.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_SPECS = {
    # name: (n_blobs, blob_radius_range, intensity_range, texture_seed_offset)
    "glioma":     dict(n_blobs=(1, 2), radius=(10, 18), intensity=(160, 220), irregular=True),
    "meningioma": dict(n_blobs=(1, 1), radius=(14, 22), intensity=(180, 230), irregular=False),
    "pituitary":  dict(n_blobs=(1, 1), radius=(6, 10),  intensity=(190, 240), irregular=False),
    "notumor":    dict(n_blobs=(0, 0), radius=(0, 0),   intensity=(0, 0),     irregular=False),
}


def _make_brain_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Rough elliptical 'skull + brain tissue' background with soft noise texture."""
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    margin = int(size * 0.08)
    draw.ellipse([margin, margin, size - margin, size - margin], fill=90)
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, 12, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1.2))
    return img


def _add_lesion(img: Image.Image, spec: dict, rng: np.random.Generator) -> Image.Image:
    size = img.size[0]
    n = rng.integers(spec["n_blobs"][0], spec["n_blobs"][1] + 1)
    draw = ImageDraw.Draw(img)
    for _ in range(n):
        r = rng.integers(spec["radius"][0], spec["radius"][1] + 1) if spec["radius"][1] > 0 else 0
        if r == 0:
            continue
        cx = rng.integers(size // 4, 3 * size // 4)
        cy = rng.integers(size // 4, 3 * size // 4)
        intensity = int(rng.integers(spec["intensity"][0], spec["intensity"][1] + 1))
        if spec["irregular"]:
            # Irregular blob: several overlapping ellipses (glioma-like diffuse shape)
            for _ in range(3):
                jx, jy = rng.integers(-r // 2, r // 2 + 1, size=2)
                jr = int(r * rng.uniform(0.6, 1.0))
                draw.ellipse([cx + jx - jr, cy + jy - jr, cx + jx + jr, cy + jy + jr], fill=intensity)
        else:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=intensity)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def generate_synthetic_dataset(out_dir: str, image_size: int = 96, n_per_class: int = 60,
                                n_patients_per_class: int = 15, seed: int = 42) -> dict:
    """
    Generates synthetic images under out_dir/<class>/patientXXXX_imgYYY.png so the
    patient-level splitter has real patient-ID structure to work with (multiple
    images share a patient prefix, mirroring the multi-slice-per-patient case the
    spec asks us to handle correctly).
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    counts = {}

    for cls_name, spec in CLASS_SPECS.items():
        cls_dir = out_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        counts[cls_name] = 0
        imgs_per_patient = max(1, n_per_class // n_patients_per_class)
        img_idx = 0
        for patient_i in range(n_patients_per_class):
            for slice_j in range(imgs_per_patient):
                img = _make_brain_background(image_size, rng)
                img = _add_lesion(img, spec, rng)
                img = img.convert("RGB")
                fname = f"patient{patient_i:04d}_img{slice_j:03d}.png"
                img.save(cls_dir / fname)
                img_idx += 1
                counts[cls_name] += 1

    return {"out_dir": str(out_path), "counts": counts, "note": "SYNTHETIC PLACEHOLDER DATA - not real MRI"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--n_per_class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate_synthetic_dataset(args.out, args.image_size, args.n_per_class, seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()
