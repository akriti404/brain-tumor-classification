"""Generate publication figures from the matched official-test comparison."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)

from data.dataset import collect_samples, patient_level_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_RESULTS = RESULTS / "epochs50"
PREDICTIONS = RUN_RESULTS / "predictions" / "unique_test"
FIGURES = RESULTS / "figures" / "journal"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEEDS = [42]
MODELS = {
    "classical_proposed": "Classical",
    "hybrid": "Hybrid CNN–VQC",
}
COLORS = {"classical_proposed": "#4C72B0", "hybrid": "#DD8452"}
METRICS = {
    "accuracy": "Accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
    "f1_weighted": "Weighted F1",
    "roc_auc_ovr": "Macro OvR ROC AUC",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _key_from_prediction_path(path: str) -> str:
    parts = Path(path.replace("\\", "/")).parts
    for index, part in enumerate(parts):
        if part.lower() in CLASSES:
            return "\\".join(parts[index:]).lower()
    raise ValueError(f"Prediction path does not contain a recognized class: {path}")


def _load_clean_predictions(model: str, seed: int, retained_paths: set[str]) -> dict:
    path = PREDICTIONS / f"{model}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-image predictions: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        by_path = {
            _key_from_prediction_path(row["image_path"]): row
            for row in csv.DictReader(file)
        }
    missing = retained_paths.difference(by_path)
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} retained test images")

    keys = sorted(retained_paths)
    rows = [by_path[key] for key in keys]
    true_labels = np.asarray([row["true_class"] for row in rows])
    predictions = np.asarray([row["predicted_class"] for row in rows])
    probabilities = np.asarray([
        [float(row[f"probability_{name}"]) for name in CLASSES]
        for row in rows
    ])
    return {
        "keys": keys,
        "y_true": true_labels,
        "y_pred": predictions,
        "y_prob": probabilities,
    }


def _metrics(result: dict) -> dict:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": float(np.average(f1, weights=[
            np.count_nonzero(y_true == name) for name in CLASSES
        ])),
        "roc_auc_ovr": roc_auc_score(
            y_true, y_prob, labels=CLASSES, multi_class="ovr", average="macro"
        ),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
    }


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_split_counts() -> dict:
    with (RUN_RESULTS / "run_config.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    samples, classes = collect_samples(cfg["data"]["root"])
    train, val, internal_test, split_method = patient_level_split(
        samples,
        cfg["data"]["val_frac"],
        cfg["data"]["test_frac"],
        42,
        cfg["data"].get("patient_level_split", True),
    )
    if split_method != "stratified_image_level":
        raise ValueError(f"Unexpected train/validation split method: {split_method}")

    def class_counts(subset: list) -> dict:
        return dict(Counter(classes[sample[1]] for sample in subset))

    return {
        "Training subset": class_counts(train),
        "Validation": class_counts(val),
        "Internal holdout": class_counts(internal_test),
    }


def _plot_dataset_splits(split_counts: dict, clean_counts: dict, full_counts: dict) -> None:
    groups = list(split_counts)
    group_counts = [split_counts[name] for name in groups]
    groups += ["Official test\n(all files)", "Test after\nexact-dedup audit"]
    group_counts += [full_counts, clean_counts]

    x = np.arange(len(CLASSES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, (name, counts) in enumerate(zip(groups, group_counts)):
        values = [counts.get(class_name, 0) for class_name in CLASSES]
        bars = ax.bar(x + (index - 2) * width, values, width, label=name)
        if index in (0, 1, 3, 4):
            ax.bar_label(bars, padding=1, fontsize=6, rotation=90)
    ax.set_xticks(x, [name.replace("_", " ").title() for name in CLASSES])
    ax.set_ylabel("Images")
    ax.set_title("Dataset split counts by class")
    ax.set_ylim(0, max(max(counts.values(), default=0) for counts in group_counts) * 1.12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095),
        frameon=False, ncol=3, fontsize=8, columnspacing=1.5,
    )
    fig.text(
        0.5, 0.025,
        "Internal holdout is reserved by the project split. Model comparison uses the audited, "
        "unique, train-disjoint official-test cohort.",
        ha="center", va="bottom", fontsize=7,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.30)
    _save_figure(fig, "fig01_dataset_split_counts")


def _plot_training_curves() -> None:
    classical = _read_json(RUN_RESULTS / "logs" / "train_classical_proposed_seed42.json")
    hybrid = _read_json(RUN_RESULTS / "logs" / "train_hybrid_seed42.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))

    for model, log in (("classical_proposed", classical), ("hybrid", hybrid)):
        history = log["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        color = COLORS[model]
        axes[0].plot(epochs, history["train_loss"], color=color, label=f"{MODELS[model]} train")
        axes[0].plot(epochs, history["val_loss"], color=color, linestyle="--",
                     label=f"{MODELS[model]} validation")
        axes[1].plot(epochs, history["train_acc"], color=color, label=f"{MODELS[model]} train")
        axes[1].plot(epochs, history["val_acc"], color=color, linestyle="--",
                     label=f"{MODELS[model]} validation")

    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Training and validation curves (50-epoch maximum, seed 42)")
    fig.tight_layout()
    _save_figure(fig, "fig02_training_curves_seed42")


def _summarize_seeds(results: dict) -> pd.DataFrame:
    rows = []
    for model, seed_results in results.items():
        for metric in METRICS:
            values = np.asarray([_metrics(item)[metric] for item in seed_results.values()])
            n = len(values)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else None
            if n > 1:
                half_width = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
                ci_lower, ci_upper = mean - half_width, mean + half_width
            else:
                ci_lower = ci_upper = None
            rows.append({
                "model": model,
                "metric": metric,
                "n_seeds": n,
                "mean": mean,
                "sample_sd": sd,
                "ci95_lower_across_seeds": ci_lower,
                "ci95_upper_across_seeds": ci_upper,
                "seed_values": ";".join(
                    f"{seed}:{_metrics(seed_results[seed])[metric]:.8f}"
                    for seed in seed_results
                ),
            })
    return pd.DataFrame(rows)


def _plot_main_comparison(results: dict, summary: pd.DataFrame) -> None:
    metrics = ["accuracy", "f1_macro"]
    labels = ["Accuracy", "Macro F1"]
    x = np.arange(len(metrics))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for index, model in enumerate(("classical_proposed", "hybrid")):
        model_summary = summary[summary["model"] == model].set_index("metric")
        means = [model_summary.loc[metric, "mean"] for metric in metrics]
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions, means, width, color=COLORS[model], label=MODELS[model],
        )
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=3, fontsize=8)
        model_results = results[model]
        for metric_index, metric in enumerate(metrics):
            vals = [ _metrics(run)[metric] for run in model_results.values() ]
            ax.scatter(
                np.full(len(vals), positions[metric_index]),
                vals, s=18, color="black", zorder=3,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Matched official-test performance (seed 42)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.01, -0.19,
        "Both models were trained with a 50-epoch maximum. Only seed 42 is available under this protocol; "
        "across-seed uncertainty is not estimable and is not shown.",
        transform=ax.transAxes, fontsize=8, va="top",
    )
    fig.tight_layout()
    _save_figure(fig, "fig03_primary_comparison")


def _plot_confusion_matrices(seed42: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, model in zip(axes, ("classical_proposed", "hybrid")):
        result = seed42[model]
        cm = confusion_matrix(result["y_true"], result["y_pred"], labels=CLASSES)
        row_totals = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set(
            xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
            xticklabels=[name.title() for name in CLASSES],
            yticklabels=[name.title() for name in CLASSES],
            xlabel="Predicted class", ylabel="True class",
            title=f"{MODELS[model]} (seed 42)",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                ax.text(
                    col, row, f"{cm[row, col]}\n{normalized[row, col]:.1%}",
                    ha="center", va="center",
                    color="white" if normalized[row, col] > 0.55 else "black",
                    fontsize=7,
                )
    fig.colorbar(image, ax=axes, shrink=0.78, pad=0.04,
                 label="Recall within true class")
    fig.suptitle("Row-normalized confusion matrices on the deduplicated test cohort")
    _save_figure(fig, "fig04_confusion_matrices_seed42")


def _plot_per_class(seed42: dict) -> pd.DataFrame:
    rows = []
    x = np.arange(len(CLASSES))
    width = 0.13
    fig, ax = plt.subplots(figsize=(9, 4.2))
    # Group the six bars into two model sections per class.
    for model_index, model in enumerate(("classical_proposed", "hybrid")):
        metrics = _metrics(seed42[model])
        model_center = x + (model_index - 0.5) * 3 * width
        for metric_name, key in (("Precision", "precision_per_class"),
                                 ("Recall", "recall_per_class"), ("F1", "f1_per_class")):
            values = metrics[key]
            bars = ax.bar(
                model_center + (("Precision", -1) if metric_name == "Precision" else
                                ("Recall", 0) if metric_name == "Recall" else ("F1", 1))[1] * width,
                values, width, color=COLORS[model],
                alpha={"Precision": 0.55, "Recall": 0.78, "F1": 1.0}[metric_name],
                label=f"{MODELS[model]} {metric_name}",
            )
            for class_index, value in enumerate(values):
                rows.append({
                    "model": model, "seed": 42, "class": CLASSES[class_index],
                    "precision": metrics["precision_per_class"][class_index],
                    "recall": metrics["recall_per_class"][class_index],
                    "f1": metrics["f1_per_class"][class_index],
                    "support": int(np.count_nonzero(seed42[model]["y_true"] == CLASSES[class_index])),
                })
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                         padding=1, fontsize=5, rotation=90)
    ax.set_xticks(x, [name.title() for name in CLASSES])
    ax.set_ylim(0.65, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall, and F1 (seed 42)")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "fig05_per_class_metrics_seed42")
    return pd.DataFrame(rows).drop_duplicates(subset=["model", "class"])


def _plot_roc(seed42: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
    for class_index, (ax, class_name) in enumerate(zip(axes.flat, CLASSES)):
        for model in ("classical_proposed", "hybrid"):
            result = seed42[model]
            binary_true = (result["y_true"] == class_name).astype(int)
            fpr, tpr, _ = roc_curve(binary_true, result["y_prob"][:, class_index])
            ax.plot(
                fpr, tpr, color=COLORS[model],
                label=f"{MODELS[model]} (AUC={auc(fpr, tpr):.3f})",
            )
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
        ax.set(
            title=class_name.title(), xlabel="False-positive rate",
            ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("One-vs-rest ROC curves on the deduplicated test cohort (seed 42)")
    fig.tight_layout()
    _save_figure(fig, "fig06_roc_curves_seed42")


def _plot_efficiency(seed42: dict) -> None:
    train = {
        model: _read_json(RUN_RESULTS / "logs" / f"train_{model}_seed42.json")
        for model in ("classical_proposed", "hybrid")
    }
    eval_files = {
        model: RUN_RESULTS / "logs" / f"eval_{model}_seed42.json"
        for model in ("classical_proposed", "hybrid")
    }
    evaluation_seconds = {model: _read_json(path)["inference_time_sec"] for model, path in eval_files.items()}
    samples = {model: _read_json(path)["n_test_samples"] for model, path in eval_files.items()}

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    models = ("classical_proposed", "hybrid")
    names = [
        f"{MODELS[model]}\n{train[model]['param_report']['total_params'] / 1_000_000:.3f}M params"
        for model in models
    ]
    accuracy = [_metrics(seed42[model])["accuracy"] for model in models]
    bars = axes[0].bar(names, accuracy, color=[COLORS[m] for m in models])
    axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in accuracy],
                      padding=3, fontsize=8)
    axes[0].set(title="Accuracy and parameter count", ylabel="Accuracy on clean test cohort",
                ylim=(0, 1))
    for axis, values, title, unit in (
        (axes[1], [train[m]["training_time_sec"] / 60 for m in models],
         "Training time", "Minutes"),
        (axes[2], [evaluation_seconds[m] / samples[m] * 1000 for m in models],
         "Evaluation time per image", "Milliseconds / image"),
    ):
        bars = axis.bar(names, values, color=[COLORS[m] for m in models])
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
        axis.set(title=title, ylabel=unit)
        axis.set_xticks(np.arange(len(models)), [MODELS[model] for model in models])
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.2)
    axes[0].grid(alpha=0.2)
    fig.suptitle("Model performance and computational cost (seed 42)")
    fig.text(
        0.5, -0.01,
        "Single-run timings. Evaluation timing covers the same audited test cohort for both models, including data loading/transforms.",
        ha="center", fontsize=7,
    )
    fig.tight_layout()
    _save_figure(fig, "fig07_efficiency_seed42")


def _box(ax, xy, width, height, text, facecolor, fontsize=8):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1, edgecolor="#333333", facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def _plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.02, 0.42), 0.12, 0.16, "MRI image", "#E8E8E8")
    _box(ax, (0.20, 0.42), 0.18, 0.16, "Pretrained\nMobileNetV2", "#C6DBEF")
    _box(ax, (0.44, 0.42), 0.16, 0.16, "Feature reducer\n1,280 → 8", "#D9EAD3")
    ax.annotate("", (0.20, 0.50), (0.14, 0.50), arrowprops={"arrowstyle": "->"})
    ax.annotate("", (0.44, 0.50), (0.38, 0.50), arrowprops={"arrowstyle": "->"})
    _box(ax, (0.68, 0.68), 0.14, 0.16, "Classical\nclassification head", "#C6DBEF")
    _box(ax, (0.66, 0.22), 0.17, 0.20, "8-qubit VQC\n3 variational layers\n24 trainable parameters", "#FCE5CD", fontsize=7)
    _box(ax, (0.85, 0.28), 0.13, 0.20, "Concat reduced\nfeatures + quantum\nexpectations;\nclassifier", "#F4CCCC", fontsize=7)
    _box(ax, (0.87, 0.68), 0.11, 0.16, "Class\nprediction", "#E8E8E8", fontsize=7)
    ax.annotate("", (0.68, 0.76), (0.60, 0.52), arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.15"})
    ax.annotate("", (0.66, 0.32), (0.60, 0.46), arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.15"})
    ax.annotate("", (0.85, 0.37), (0.83, 0.32), arrowprops={"arrowstyle": "->"})
    ax.annotate("", (0.85, 0.43), (0.60, 0.45),
                arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.28"})
    ax.annotate("", (0.87, 0.76), (0.82, 0.76), arrowprops={"arrowstyle": "->"})
    ax.annotate("", (0.92, 0.68), (0.92, 0.48),
                arrowprops={"arrowstyle": "->"})
    ax.text(0.5, 0.94, "Classical control and hybrid CNN–VQC comparison",
            ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.08,
            "Both models share the same CNN backbone and reducer; the hybrid path adds quantum expectations to the reduced features.",
            ha="center", fontsize=8)
    _save_figure(fig, "fig08_architecture_comparison")


def _write_notes_and_tables(
    results: dict,
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    split_counts: dict,
    clean_counts: dict,
    full_counts: dict,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables_dir = RUN_RESULTS / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "journal_seed_summary.csv", index=False)
    per_class.to_csv(tables_dir / "journal_seed42_per_class.csv", index=False)
    split_rows = [
        {"subset": subset, "class": class_name, "count": counts.get(class_name, 0)}
        for subset, counts in (
            *split_counts.items(),
            ("Official test (all files)", full_counts),
            ("Official test (audited cohort)", clean_counts),
        )
        for class_name in CLASSES
    ]
    pd.DataFrame(split_rows).to_csv(
        tables_dir / "journal_dataset_split_counts.csv", index=False
    )

    seed42 = {model: results[model][42] for model in results}
    metrics = {model: _metrics(result) for model, result in seed42.items()}
    train_logs = {
        model: _read_json(RUN_RESULTS / "logs" / f"train_{model}_seed42.json")
        for model in ("classical_proposed", "hybrid")
    }
    eval_logs = {
        model: _read_json(RUN_RESULTS / "logs" / f"eval_{model}_seed42.json")
        for model in ("classical_proposed", "hybrid")
    }
    class_auc = {
        model: {
            name: roc_auc_score(
                (result["y_true"] == name).astype(int),
                result["y_prob"][:, class_index],
            )
            for class_index, name in enumerate(CLASSES)
        }
        for model, result in seed42.items()
    }
    class_index = {name: index for index, name in enumerate(CLASSES)}
    matrices = {
        model: confusion_matrix(
            result["y_true"], result["y_pred"], labels=CLASSES
        )
        for model, result in seed42.items()
    }
    meningioma_support = int(matrices["classical_proposed"][class_index["meningioma"]].sum())
    meningioma_to_glioma = {
        model: int(matrix[class_index["meningioma"], class_index["glioma"]])
        for model, matrix in matrices.items()
    }
    training_minutes = {
        model: train_logs[model]["training_time_sec"] / 60
        for model in train_logs
    }
    inference_ms_per_image = {
        model: eval_logs[model]["inference_time_sec"] / eval_logs[model]["n_test_samples"] * 1000
        for model in eval_logs
    }
    clean_total = sum(clean_counts.values())
    raw_total = sum(full_counts.values())
    classical = metrics["classical_proposed"]
    hybrid = metrics["hybrid"]
    class_auc_text = "; ".join(
        f"{name}: {class_auc['hybrid'][name]:.3f}" for name in CLASSES
    )
    notes = [
        "# Journal figures: captions and interpretation",
        "",
        f"All model figures use seed-42 runs with a 50-epoch maximum and validation-accuracy early stopping. The primary comparison uses {clean_total:,} unique, train-disjoint images from the separate official Kaggle test folder, evaluated identically for both models. Earlier runs with a 30-epoch maximum are not pooled into this comparison.",
        "",
        "## Figures and findings",
        "",
        f"1. **Dataset split counts** (`fig01_dataset_split_counts`): the project training dataset has {sum(split_counts['Training subset'].values()):,} train, {sum(split_counts['Validation'].values()):,} validation, and {sum(split_counts['Internal holdout'].values()):,} internal holdout images. The separate official test folder contains {raw_total:,} files; the audited evaluation cohort contains {clean_total:,}. Class-wise counts are in `results/epochs50/tables/journal_dataset_split_counts.csv`.",
        f"2. **Training curves** (`fig02_training_curves_seed42`): best validation accuracy was {train_logs['classical_proposed']['best_val_acc']:.1%} for classical and {train_logs['hybrid']['best_val_acc']:.1%} for hybrid. They completed {len(train_logs['classical_proposed']['history']['train_loss'])} and {len(train_logs['hybrid']['history']['train_loss'])} epochs, respectively. The curves show optimization progress, not test-set superiority.",
        f"3. **Primary performance** (`fig03_primary_comparison`): on the matched cohort, classical accuracy/macro F1 were {classical['accuracy']:.1%}/{classical['f1_macro']:.1%}; hybrid accuracy/macro F1 were {hybrid['accuracy']:.1%}/{hybrid['f1_macro']:.1%}. Only seed 42 was retrained under the 50-epoch protocol, so across-seed uncertainty is not estimable and is not shown.",
        f"4. **Row-normalized confusion matrices** (`fig04_confusion_matrices_seed42`): rows show recall within the true class; each cell reports count and row percentage. Meningioma-to-glioma errors were {meningioma_to_glioma['classical_proposed']}/{meningioma_support} ({meningioma_to_glioma['classical_proposed'] / meningioma_support:.1%}) for classical and {meningioma_to_glioma['hybrid']}/{meningioma_support} ({meningioma_to_glioma['hybrid'] / meningioma_support:.1%}) for hybrid. These are single-seed descriptive results.",
        "5. **Per-class metrics** (`fig05_per_class_metrics_seed42`): compare precision, recall, and F1 across all four classes to reveal class-specific trade-offs, including glioma/meningioma confusions that aggregate metrics can conceal.",
        f"6. **One-vs-rest ROC curves** (`fig06_roc_curves_seed42`): curves show class-wise discrimination over thresholds on the audited cohort. Macro OvR AUC was {classical['roc_auc_ovr']:.3f} for classical and {hybrid['roc_auc_ovr']:.3f} for hybrid. Hybrid class-wise AUCs (glioma, meningioma, no-tumor, pituitary) were {class_auc_text}; AUC does not replace thresholded metrics or calibration analysis.",
        f"7. **Efficiency comparison** (`fig07_efficiency_seed42`): parameter counts were {train_logs['classical_proposed']['param_report']['total_params']:,} for classical and {train_logs['hybrid']['param_report']['total_params']:,} for hybrid. Training took {training_minutes['classical_proposed']:.2f} and {training_minutes['hybrid']:.2f} minutes; evaluation averaged {inference_ms_per_image['classical_proposed']:.2f} and {inference_ms_per_image['hybrid']:.2f} ms/image, respectively. These are single-run measurements; evaluation time includes test data loading/transforms.",
        "8. **Architecture schematic** (`fig08_architecture_comparison`): both models share the MobileNetV2 feature extractor and reducer. The hybrid path adds a VQC; its measured expectations are combined with reduced classical features before classification.",
        "9. **Paired explainability panels**: four separate class-specific images use identical audited test images for both models, with up to four cases per class: high-confidence hybrid-correct examples and one error example. They use the epoch-50 checkpoints and prioritize glioma/meningioma confusions. The dataset has no tumor masks or boxes, so these are qualitative illustrations rather than validated lesion localization; hybrid Grad-CAM reflects CNN-backbone gradients, not the VQC's internal computation.",
        "",
        "## Statistical and reporting cautions",
        "",
        "- Each model has one seed under the 50-epoch protocol. Figure 3 therefore has no across-seed standard deviation or confidence interval; do not infer statistical significance.",
        f"- The raw official test folder contains {raw_total:,} files; the primary comparison uses the {clean_total:,}-image audited, unique, train-disjoint cohort.",
        "- The internal holdout documents the configured project split; reported model comparison uses the separate official Kaggle test cohort.",
    ]
    (FIGURES / "figure_captions_and_findings.md").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )


def generate() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit_path = RESULTS / "tables" / "official_test_integrity_audit.csv"
    with audit_path.open(newline="", encoding="utf-8") as file:
        audit = list(csv.DictReader(file))
    retained = {
        row["image_path"].replace("/", "\\").lower()
        for row in audit if row["status"] == "retained_unique"
    }
    if not retained:
        raise ValueError(f"No retained test images found in {audit_path}")

    results = {
        "classical_proposed": {
            42: _load_clean_predictions("classical_proposed", 42, retained),
        },
        "hybrid": {
            seed: _load_clean_predictions("hybrid", seed, retained)
            for seed in SEEDS
        },
    }
    reference_keys = results["hybrid"][42]["keys"]
    reference_labels = results["hybrid"][42]["y_true"]
    for model_runs in results.values():
        for seed, result in model_runs.items():
            if result["keys"] != reference_keys or not np.array_equal(result["y_true"], reference_labels):
                raise ValueError(f"{model_runs} seed {seed} does not use the same test cohort")

    split_counts = _load_split_counts()
    full_samples, _ = collect_samples(str(ROOT / "data" / "kaggle_test"))
    full_counts = dict(Counter(CLASSES[sample[1]] for sample in full_samples))
    clean_counts = dict(Counter(Path(name).parts[0] for name in retained))
    seed42 = {model: runs[42] for model, runs in results.items()}

    _plot_dataset_splits(split_counts, clean_counts, full_counts)
    _plot_training_curves()
    summary = _summarize_seeds(results)
    _plot_main_comparison(results, summary)
    _plot_confusion_matrices(seed42)
    per_class = _plot_per_class(seed42)
    _plot_roc(seed42)
    _plot_efficiency(seed42)
    _plot_architecture()
    _write_notes_and_tables(results, summary, per_class, split_counts, clean_counts, full_counts)
    print(f"Publication figures and notes saved in {FIGURES}")


if __name__ == "__main__":
    generate()
