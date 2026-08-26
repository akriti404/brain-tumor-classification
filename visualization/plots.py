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
import numpy as np
import pandas as pd
import seaborn as sns
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


def plot_resource_ablations(results_csv_path: str, out_path: str):
    """Plot resource ablation results showing impact of qubits, layers, and re-uploading."""
    df = pd.read_csv(results_csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Qubit count vs accuracy
    if 'n_qubits' in df.columns:
        for representation in df['representation'].unique():
            rep_df = df[df['representation'] == representation]
            qubit_means = rep_df.groupby('n_qubits')['accuracy'].mean()
            qubit_stds = rep_df.groupby('n_qubits')['accuracy'].std()
            axes[0, 0].errorbar(qubit_means.index, qubit_means.values, 
                              yerr=qubit_stds.values, marker='o', 
                              label=representation, capsize=5)
        axes[0, 0].set_xlabel('Number of Qubits')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs. Qubit Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Layer count vs accuracy
    if 'n_layers' in df.columns:
        for representation in df['representation'].unique():
            rep_df = df[df['representation'] == representation]
            layer_means = rep_df.groupby('n_layers')['accuracy'].mean()
            layer_stds = rep_df.groupby('n_layers')['accuracy'].std()
            axes[0, 1].errorbar(layer_means.index, layer_means.values, 
                              yerr=layer_stds.values, marker='s', 
                              label=representation, capsize=5)
        axes[0, 1].set_xlabel('Number of Layers')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs. Layer Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Re-uploading comparison
    if 'data_reuploading' in df.columns:
        reupload_data = []
        for representation in df['representation'].unique():
            for reupload in [True, False]:
                rep_df = df[(df['representation'] == representation) & 
                           (df['data_reuploading'] == reupload)]
                if len(rep_df) > 0:
                    reupload_data.append({
                        'representation': representation,
                        'reuploading': 'Yes' if reupload else 'No',
                        'accuracy': rep_df['accuracy'].mean(),
                        'std': rep_df['accuracy'].std()
                    })
        
        if reupload_data:
            reupload_df = pd.DataFrame(reupload_data)
            x_pos = np.arange(len(reupload_df['representation'].unique()))
            width = 0.35
            
            for i, representation in enumerate(reupload_df['representation'].unique()):
                rep_data = reupload_df[reupload_df['representation'] == representation]
                for j, reupload in enumerate(['Yes', 'No']):
                    data = rep_data[rep_data['reuploading'] == reupload]
                    if len(data) > 0:
                        axes[1, 0].bar(x_pos[i] + j*width, data['accuracy'].values[0], 
                                     width, yerr=data['std'].values[0] if data['std'].values[0] > 0 else None,
                                     label=f'{representation} - {reupload}', capsize=5)
            
            axes[1, 0].set_xlabel('Representation')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Re-uploading Impact')
            axes[1, 0].set_xticks(x_pos + width/2)
            axes[1, 0].set_xticklabels(reupload_df['representation'].unique())
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # Combined heatmap
    if 'n_qubits' in df.columns and 'n_layers' in df.columns:
        pivot_data = df.groupby(['n_qubits', 'n_layers', 'representation'])['accuracy'].mean().unstack()
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[1, 1], cbar_kws={'label': 'Accuracy'})
            axes[1, 1].set_title('Accuracy Heatmap (Qubits × Layers)')
            axes[1, 1].set_xlabel('Number of Layers')
            axes[1, 1].set_ylabel('Number of Qubits')
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_noise_experiments(results_csv_path: str, out_path: str):
    """Plot noise experiment results showing impact of different noise types and probabilities."""
    df = pd.read_csv(results_csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Noise type comparison
    if 'noise_type' in df.columns:
        for representation in df['representation'].unique():
            rep_df = df[df['representation'] == representation]
            noise_means = rep_df.groupby('noise_type')['accuracy'].mean()
            noise_stds = rep_df.groupby('noise_type')['accuracy'].std()
            axes[0, 0].bar(noise_means.index, noise_means.values, 
                         yerr=noise_stds.values, alpha=0.7, 
                         label=representation, capsize=5)
        axes[0, 0].set_xlabel('Noise Type')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy by Noise Type')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Noise probability impact
    if 'noise_prob' in df.columns:
        for representation in df['representation'].unique():
            rep_df = df[df['representation'] == representation]
            prob_means = rep_df.groupby('noise_prob')['accuracy'].mean()
            prob_stds = rep_df.groupby('noise_prob')['accuracy'].std()
            axes[0, 1].errorbar(prob_means.index, prob_means.values, 
                              yerr=prob_stds.values, marker='o', 
                              label=representation, capsize=5)
        axes[0, 1].set_xlabel('Noise Probability')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs. Noise Probability')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Combined noise type and probability
    if 'noise_type' in df.columns and 'noise_prob' in df.columns:
        for noise_type in df['noise_type'].unique():
            noise_df = df[df['noise_type'] == noise_type]
            prob_means = noise_df.groupby('noise_prob')['accuracy'].mean()
            prob_stds = noise_df.groupby('noise_prob')['accuracy'].std()
            axes[1, 0].errorbar(prob_means.index, prob_means.values, 
                              yerr=prob_stds.values, marker='o', 
                              label=noise_type, capsize=5)
        axes[1, 0].set_xlabel('Noise Probability')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Noise Impact by Type and Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Robustness analysis (ideal vs noisy)
    if 'noise_type' in df.columns:
        ideal_df = df[df['noise_type'] == 'ideal']
        noisy_df = df[df['noise_type'] != 'ideal']
        
        if len(ideal_df) > 0 and len(noisy_df) > 0:
            ideal_acc = ideal_df['accuracy'].mean()
            noisy_acc = noisy_df['accuracy'].mean()
            
            labels = ['Ideal', 'Noisy']
            accuracies = [ideal_acc, noisy_acc]
            
            axes[1, 1].bar(labels, accuracies, alpha=0.7, 
                         color=['green', 'red'])
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Robustness: Ideal vs. Noisy')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_statistical_analysis(results_json_path: str, out_path: str):
    """Plot statistical analysis results including confidence intervals and effect sizes."""
    with open(results_json_path) as f:
        data = json.load(f)
    
    analyses = data.get('analyses', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Ablation trends
    if 'ablation_cnn_hybrid' in analyses or 'ablation_gnn_hybrid' in analyses:
        for key in analyses:
            if key.startswith('ablation_'):
                ablation_data = analyses[key]
                
                # Qubit effects
                if 'qubit_effects' in ablation_data:
                    qubit_data = ablation_data['qubit_effects']
                    qubits = sorted([int(k) for k in qubit_data.keys()])
                    means = [qubit_data[str(q)]['mean'] for q in qubits]
                    cis = [(qubit_data[str(q)]['ci_upper'] - qubit_data[str(q)]['ci_lower'])/2 
                           for q in qubits]
                    
                    axes[0, 0].errorbar(qubits, means, yerr=cis, marker='o', 
                                       label=key.replace('ablation_', ''), capsize=5)
        
        axes[0, 0].set_xlabel('Number of Qubits')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Qubit Count Effects with CI')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Noise effects
    noise_keys = [k for k in analyses.keys() if k.startswith('noise_')]
    if noise_keys:
        for key in noise_keys:
            noise_data = analyses[key]
            
            # Extract noise effects
            if isinstance(noise_data, dict):
                for noise_key, stats in noise_data.items():
                    if 'mean' in stats:
                        representation = key.replace('noise_', '')
                        axes[0, 1].bar(noise_key, stats['mean'], 
                                     yerr=stats['std'] if 'std' in stats else None,
                                     alpha=0.7, label=representation, capsize=5)
        
        axes[0, 1].set_xlabel('Noise Configuration')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Noise Effects on Accuracy')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Effect sizes (if available)
    effect_size_keys = [k for k in analyses.keys() if 'effect_size' in analyses.get(k, {})]
    if effect_size_keys:
        for key in effect_size_keys:
            effect_data = analyses[key].get('effect_size', {})
            if 'value' in effect_data:
                axes[1, 0].bar(key, effect_data['value'], alpha=0.7)
                axes[1, 0].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Small effect')
                axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
                axes[1, 0].axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Large effect')
        
        axes[1, 0].set_ylabel("Cohen's d")
        axes[1, 0].set_title('Effect Sizes')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    if 'config_summary' in data:
        config = data['config_summary']
        seeds = config.get('seeds_for_stats', [])
        if seeds:
            axes[1, 1].text(0.5, 0.5, f'Statistical Analysis Summary\n\nSeeds used: {seeds}\nAnalyses performed: {len(analyses)}', 
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Analysis Summary')
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_member3_figures(results_dir: str = "results"):
    """Generate all Member 3 experimental figures."""
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Member 3 experimental figures...")
    
    # Resource ablation plots
    ablation_csv = Path(results_dir) / "tables" / "resource_ablations.csv"
    if ablation_csv.exists():
        plot_resource_ablations(str(ablation_csv), str(fig_dir / "06_resource_ablations.png"))
        print(f"Generated resource ablation plot")
    
    # Noise experiment plots
    noise_csv = Path(results_dir) / "tables" / "noise_experiments.csv"
    if noise_csv.exists():
        plot_noise_experiments(str(noise_csv), str(fig_dir / "07_noise_experiments.png"))
        print(f"Generated noise experiments plot")
    
    # Statistical analysis plots
    stats_json = Path(results_dir) / "statistical_analysis" / "statistical_analysis_report.json"
    if stats_json.exists():
        plot_statistical_analysis(str(stats_json), str(fig_dir / "08_statistical_analysis.png"))
        print(f"Generated statistical analysis plot")
    
    print(f"Member 3 figures written to {fig_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Data root directory")
    parser.add_argument("--member3", action="store_true", help="Generate Member 3 experimental figures")
    args = parser.parse_args()
    
    generate_core_figures(args.results_dir, args.data_root)
    if args.member3:
        generate_member3_figures(args.results_dir)
