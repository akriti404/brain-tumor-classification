"""
Dataset inspection.

Given an ImageFolder-style directory (root/<class_name>/*.jpg|png), this
module automatically discovers:
  - number of classes and their names (never assumed / hard-coded)
  - number of images per class
  - image dimension distribution
  - corrupted / unreadable images
  - exact duplicate images (via content hash)

Run standalone:  python -m data.dataset_inspector --root data/raw
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_dataset(root: str) -> dict:
    root_path = Path(root)
    if not root_path.exists():
        return {"error": f"Dataset root '{root}' does not exist.", "n_classes": 0, "n_images": 0}

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    per_class_counts = {}
    dims = Counter()
    corrupted = []
    hashes = defaultdict(list)
    total_images = 0

    for cls_dir in class_dirs:
        files = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
        per_class_counts[cls_dir.name] = len(files)
        total_images += len(files)
        for f in files:
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    dims[im.size] += 1
                hashes[_file_hash(f)].append(str(f))
            except Exception as e:
                corrupted.append({"path": str(f), "error": str(e)})

    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}

    most_common_dims = dims.most_common(5)

    report = {
        "root": str(root_path),
        "n_classes": len(class_names),
        "class_names": class_names,
        "n_images": total_images,
        "class_distribution": per_class_counts,
        "class_balance_ratio": (
            round(min(per_class_counts.values()) / max(per_class_counts.values()), 3)
            if per_class_counts and max(per_class_counts.values()) > 0 else None
        ),
        "top_image_dimensions": [{"size": list(size), "count": c} for size, c in most_common_dims],
        "n_corrupted": len(corrupted),
        "corrupted_examples": corrupted[:10],
        "n_duplicate_groups": len(duplicates),
        "n_duplicate_files": sum(len(v) - 1 for v in duplicates.values()),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="results/logs/dataset_inspection.json")
    args = parser.parse_args()

    report = inspect_dataset(args.root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
