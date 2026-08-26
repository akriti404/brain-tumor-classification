"""
Core publication-style plots generated from results/tables/experiment_results.csv
and the per-model training logs. Saves everything to results/figures/.

This module covers the plots relevant to the "core pieces" scope (data,
baselines, hybrid model): dataset class distribution, sample images,
training curves, baseline comparison, and accuracy-vs-parameters. Noise,
cross-dataset, and full ablation plots are added once those experiment
stages are implemented (see README "What's implemented vs scaffolded").
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})


def plot_class_distribution(inspection_json_path: str, out_path: str):
    with open(inspection_json_path) as f:
        report = json.load(f)
    dist = report["class_distribution"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(dist.keys()), list(dist.values()), color="#4C72B0")
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Class")
    for i, v in enumerate(dist.values()):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sample_images(data_root: str, classes: list, out_path: str, n_per_class: int = 3):
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 2, len(classes) * 2))
    for row, cls in enumerate(classes):
        cls_dir = Path(data_root) / cls
        files = sorted(cls_dir.glob("*"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row, col] if len(classes) > 1 else axes[col]
            if col < len(files):
                img = Image.open(files[col])
                ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=9)
        axes[row, 0].set_title(cls, fontsize=10, loc="left")
    fig.suptitle("Sample MRI Images per Class")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_curves(train_log_json_path: str, out_path: str):
    with open(train_log_json_path) as f:
        log = json.load(f)
    hist = log["history"]
    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, hist["train_loss"], label="train", marker="o")
    axes[0].plot(epochs, hist["val_loss"], label="val", marker="o")
    axes[0].set_title(f"{log['model_name']} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, hist["train_acc"], label="train", marker="o")
    axes[1].plot(epochs, hist["val_acc"], label="val", marker="o")
    axes[1].set_title(f"{log['model_name']} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_baseline_comparison(results_csv_path: str, out_path: str):
    df = pd.read_csv(results_csv_path)
    df = df.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(df["model"], df["accuracy"], color="#55A868")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Baseline vs. Proposed Model Comparison")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_accuracy_vs_parameters(results_csv_path: str, out_path: str):
    df = pd.read_csv(results_csv_path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["total_params"], df["accuracy"], s=80, color="#C44E52")
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["total_params"], row["accuracy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Total trainable parameters (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. Parameter Count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_core_figures(results_dir: str = "results", data_root: str = "data/raw"):
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inspection_path = Path(results_dir) / "logs" / "dataset_inspection.json"
    if inspection_path.exists():
        plot_class_distribution(str(inspection_path), str(fig_dir / "01_class_distribution.png"))
        with open(inspection_path) as f:
            classes = json.load(f)["class_names"]
        plot_sample_images(data_root, classes, str(fig_dir / "02_sample_images.png"))

    for log_file in sorted((Path(results_dir) / "logs").glob("train_*.json")):
        name = log_file.stem.replace("train_", "")
        plot_training_curves(str(log_file), str(fig_dir / f"03_training_curves_{name}.png"))

    results_csv = Path(results_dir) / "tables" / "experiment_results.csv"
    if results_csv.exists():
        plot_baseline_comparison(str(results_csv), str(fig_dir / "04_baseline_comparison.png"))
        plot_accuracy_vs_parameters(str(results_csv), str(fig_dir / "05_accuracy_vs_parameters.png"))

    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    generate_core_figures()
