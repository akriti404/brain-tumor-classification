"""
Resource ablation experiments for quantum resource analysis.

Systematically evaluates the impact of quantum resources (qubits, layers, re-uploading)
on performance for both CNN-VQC and GNN-VQC branches.
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_ablation_config(
    base_config: Dict,
    n_qubits: int,
    n_layers: int,
    data_reuploading: bool,
) -> Dict:
    """Create a modified config for a specific ablation setting."""
    config = copy.deepcopy(base_config)
    config["quantum"]["n_qubits"] = n_qubits
    config["quantum"]["n_layers"] = n_layers
    config["quantum"]["data_reuploading"] = data_reuploading
    return config


def save_temp_config(config: Dict, temp_path: Path) -> str:
    """Save temporary config file."""
    with open(temp_path, "w") as f:
        yaml.dump(config, f)
    return str(temp_path)


def run_single_ablation(
    model: str,
    representation: str,
    n_qubits: int,
    n_layers: int,
    data_reuploading: bool,
    base_config: Dict,
    temp_config_dir: Path,
    seed: int = 42,
) -> Dict:
    """Run a single ablation experiment."""
    
    # Create ablation-specific config
    ablation_config = create_ablation_config(
        base_config, n_qubits, n_layers, data_reuploading
    )
    
    # Save temporary config
    temp_config_path = temp_config_dir / f"temp_q{n_qubits}_l{n_layers}_ru{data_reuploading}.yaml"
    config_path = save_temp_config(ablation_config, temp_config_path)
    
    print(f"\nRunning ablation: {representation}-{model}")
    print(f"  Qubits: {n_qubits}, Layers: {n_layers}, Re-uploading: {data_reuploading}")
    print(f"  Seed: {seed}")
    
    # Training
    train_cmd = [
        sys.executable,
        "train.py",
        "--model", model,
        "--representation", representation,
        "--config", config_path,
        "--seed", str(seed),
    ]
    
    try:
        train_result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"  Training timed out")
        return None
    except Exception as e:
        print(f"  Training failed with exception: {e}")
        return None
    
    if train_result.returncode != 0:
        print(f"  Training failed: {train_result.stderr}")
        return None
    
    # Evaluation
    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--model", model,
        "--representation", representation,
        "--config", config_path,
        "--seed", str(seed),
    ]
    
    try:
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"  Evaluation timed out")
        return None
    except Exception as e:
        print(f"  Evaluation failed with exception: {e}")
        return None
    
    if eval_result.returncode != 0:
        print(f"  Evaluation failed: {eval_result.stderr}")
        return None
    
    # Load evaluation results
    results_dir = Path(base_config["project"]["results_dir"])
    eval_log_path = (
        results_dir / "logs" / 
        f"eval_{representation + '_' if representation == 'gnn' else ''}{model}_seed{seed}.json"
    )
    
    if eval_log_path.exists():
        try:
            with open(eval_log_path) as f:
                eval_log = json.load(f)
            
            # Add ablation metadata
            eval_log["ablation"] = {
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "data_reuploading": data_reuploading,
            }
            
            print(f"  Completed successfully")
            return eval_log
        except json.JSONDecodeError as e:
            print(f"  Error loading evaluation log: {e}")
            return None
    else:
        print(f"  Warning: Evaluation log not found")
        return None


def run_ablation_sweep(
    model: str,
    representation: str,
    qubit_counts: List[int],
    layer_counts: List[int],
    reuploading_options: List[bool],
    base_config: Dict,
    temp_config_dir: Path,
    seeds: List[int],
) -> List[Dict]:
    """Run a full ablation sweep for a model-representation pair."""
    
    results = []
    total_experiments = (
        len(qubit_counts) * len(layer_counts) * len(reuploading_options) * len(seeds)
    )
    
    print(f"\n{'='*60}")
    print(f"Ablation sweep for {representation.upper()}-{model}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}")
    
    experiment_count = 0
    
    for n_qubits in qubit_counts:
        for n_layers in layer_counts:
            for data_reuploading in reuploading_options:
                for seed in seeds:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments}]")
                    
                    result = run_single_ablation(
                        model=model,
                        representation=representation,
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        data_reuploading=data_reuploading,
                        base_config=base_config,
                        temp_config_dir=temp_config_dir,
                        seed=seed,
                    )
                    
                    if result is not None:
                        results.append(result)
    
    print(f"\nCompleted {len(results)}/{total_experiments} experiments")
    return results


def save_ablation_results(
    results: List[Dict],
    model: str,
    representation: str,
    results_dir: str = "results",
) -> None:
    """Save ablation results to CSV and JSON."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    json_path = results_path / "logs" / f"ablation_{representation}_{model}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved detailed results to {json_path}")
    
    # Create summary CSV
    csv_path = results_path / "tables" / "resource_ablations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "model", "representation", "seed",
        "n_qubits", "n_layers", "data_reuploading",
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted", "roc_auc_ovr", "specificity_macro",
        "total_params", "quantum_parameters", "training_time_sec", "inference_time_sec"
    ]
    
    write_header = not csv_path.exists()
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        for result in results:
            ablation = result.get("ablation", {})
            row = {
                "model": f"{representation}_{model}" if representation == "gnn" else model,
                "representation": representation,
                "seed": result.get("seed"),
                "n_qubits": ablation.get("n_qubits"),
                "n_layers": ablation.get("n_layers"),
                "data_reuploading": ablation.get("data_reuploading"),
                "accuracy": result.get("accuracy"),
                "precision_macro": result.get("precision_macro"),
                "recall_macro": result.get("recall_macro"),
                "f1_macro": result.get("f1_macro"),
                "f1_weighted": result.get("f1_weighted"),
                "roc_auc_ovr": result.get("roc_auc_ovr"),
                "specificity_macro": result.get("specificity_macro"),
                "total_params": result.get("total_params"),
                "quantum_parameters": result.get("quantum_parameters"),
                "training_time_sec": result.get("training_time_sec"),
                "inference_time_sec": result.get("inference_time_sec"),
            }
            writer.writerow(row)
    
    print(f"Updated ablation results table at {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run resource ablation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["hybrid"],
        help="Models to evaluate"
    )
    parser.add_argument(
        "--representations",
        type=str,
        nargs="+",
        choices=["cnn", "gnn"],
        default=["cnn", "gnn"],
        help="Representations to evaluate"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Qubit counts to sweep"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Layer counts to sweep"
    )
    parser.add_argument(
        "--reuploading",
        type=str,
        nargs="+",
        choices=["true", "false", "both"],
        default=["both"],
        help="Re-uploading settings (true, false, or both)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for multi-seed validation (default: use config)"
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    
    # Parse re-uploading options
    if "both" in args.reuploading:
        reuploading_options = [True, False]
    else:
        reuploading_options = [opt.lower() == "true" for opt in args.reuploading]
    
    # Get seeds
    seeds = args.seeds if args.seeds is not None else base_config["project"].get("seeds_for_stats", [42])
    
    # Create temporary config directory
    temp_config_dir = Path("temp_ablation_configs")
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Resource Ablation Experiments")
    print(f"Config: {args.config}")
    print(f"Models: {args.models}")
    print(f"Representations: {args.representations}")
    print(f"Qubits: {args.qubits}")
    print(f"Layers: {args.layers}")
    print(f"Re-uploading: {reuploading_options}")
    print(f"Seeds: {seeds}")
    
    # Run ablation sweeps
    all_results = []
    
    for model in args.models:
        for representation in args.representations:
            results = run_ablation_sweep(
                model=model,
                representation=representation,
                qubit_counts=args.qubits,
                layer_counts=args.layers,
                reuploading_options=reuploading_options,
                base_config=base_config,
                temp_config_dir=temp_config_dir,
                seeds=seeds,
            )
            
            if results:
                save_ablation_results(results, model, representation)
                all_results.extend(results)
    
    # Cleanup temporary configs
    import shutil
    if temp_config_dir.exists():
        shutil.rmtree(temp_config_dir)
        print(f"\nCleaned up temporary config directory")
    
    if all_results:
        print(f"\n{'='*60}")
        print("Resource ablation experiments completed successfully")
        print(f"Total results collected: {len(all_results)}")
        print(f"{'='*60}")
    else:
        print("\nNo successful experiments")
        sys.exit(1)


if __name__ == "__main__":
    main()