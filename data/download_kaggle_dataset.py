"""Download and organize the Kaggle Brain Tumor MRI dataset for this project.

This script uses KaggleHub to fetch the public dataset and rearranges the extracted
folders into the expected project layout:
    data/raw/<class_name>/*.jpg

Example:
    python -m data.download_kaggle_dataset --out data/raw
"""

import argparse
import shutil
from pathlib import Path

import kagglehub


def download_dataset() -> Path:
    """Download the latest Kaggle dataset snapshot and return the extracted root."""
    print("Downloading Kaggle dataset via kagglehub...")
    return Path(kagglehub.dataset_download("deeppythonist/brain-tumor-mri-dataset"))


def copy_class_folders(source_root: Path, out_root: Path, include_testing: bool = False) -> list[str]:
    """Copy the train split into data/raw/<class> and optionally the test split too."""
    out_root.mkdir(parents=True, exist_ok=True)

    # Remove stale split directories left from a previous bad copy so they do not get
    # treated as dataset classes in the loader.
    for stale_name in ("train", "test", "Training", "Testing", "training", "testing"):
        stale_dir = out_root / stale_name
        if stale_dir.exists():
            shutil.rmtree(stale_dir)

    copied = []

    def _copy_dir_tree(source_dir: Path):
        if not source_dir.exists():
            return

        for class_dir in sorted(source_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            target_dir = out_root / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for file in sorted(class_dir.iterdir()):
                if file.is_file():
                    shutil.copy2(file, target_dir / file.name)
                    count += 1
            copied.append(f"{class_dir.name}: {count} files")

    # KaggleHub commonly extracts as: root/train/<class> and root/test/<class>
    for split_name in ("train", "Training", "training"):
        _copy_dir_tree(source_root / split_name)

    if include_testing:
        for split_name in ("test", "Testing", "testing"):
            _copy_dir_tree(source_root / split_name)

    # Fallback for datasets whose top level already contains class folders directly.
    if not copied:
        for class_dir in sorted(source_root.iterdir()):
            if not class_dir.is_dir():
                continue
            target_dir = out_root / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for file in sorted(class_dir.iterdir()):
                if file.is_file():
                    shutil.copy2(file, target_dir / file.name)
                    count += 1
            copied.append(f"{class_dir.name}: {count} files")

    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/raw", help="Destination for organized class folders")
    parser.add_argument("--include-testing", action="store_true", help="Also copy the Testing split into the output")
    args = parser.parse_args()

    out_root = Path(args.out)
    dataset_root = download_dataset()
    print(f"Dataset downloaded to: {dataset_root}")

    copies = copy_class_folders(dataset_root, out_root, include_testing=args.include_testing)
    if not copies:
        raise RuntimeError(f"No class folders were found under {dataset_root}. Check the Kaggle dataset layout.")

    print("Organized dataset folders:")
    for item in copies:
        print(f"  - {item}")
    print(f"Ready for training: {out_root}")


if __name__ == "__main__":
    main()
