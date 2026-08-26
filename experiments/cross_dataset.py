"""
Cross-dataset generalization experiments.

Evaluates model robustness by training on one dataset and testing on another
to measure domain shift impact on both CNN and GNN representations.
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


def create_cross_dataset_config(
    base_config: Dict,
    source_dataset: str,
    target_dataset: str,
) -> Dict:
    """Create a modified config for cross-dataset experiments."""
    config = copy.deepcopy(base_config)
    
    # Update data root to source dataset for training
    config["data"]["root"] = source_dataset
    
    # Store target dataset for testing
    config["cross_dataset"] = {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
    }
    
    return config


def save_temp_config(config: Dict, temp_path: Path) -> str:
    """Save temporary config file."""
    with open(temp_path, "w") as f:
        yaml.dump(config, f)
    return str(temp_path)


def run_single_cross_dataset_experiment(
    model: str,
    representation: str,
    source_dataset: str,
    target_dataset: str,
    base_config: Dict,
    temp_config_dir: Path,
    seed: int = 42,
) -> Dict:
    """Run a single cross-dataset experiment."""
    
    # Create cross-dataset config
    cross_config = create_cross_dataset_config(
        base_config, source_dataset, target_dataset
    )
    
    # Save temporary config
    temp_config_path = temp_config_dir / f"temp_cross_{Path(source_dataset).name}_to_{Path(target_dataset).name}.yaml"
    config_path = save_temp_config(cross_config, temp_config_path)
    
    print(f"\nRunning cross-dataset experiment: {representation}-{model}")
    print(f"  Source: {source_dataset}")
    print(f"  Target: {target_dataset}")
    print(f"  Seed: {seed}")
    
    # Training on source dataset
    train_cmd = [
        sys.executable,
        "train.py",
        "--model", model,
        "--representation", representation,
        "--config", config_path,
        "--seed", str(seed),
    ]
    
    train_result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if train_result.returncode != 0:
        print(f"  Training failed: {train_result.stderr}")
        return None
    
    print(f"  Training on source dataset completed")
    
    # For proper cross-dataset evaluation, we need to:
    # 1. Load the trained model
    # 2. Evaluate on target dataset
    # For now, we'll do a simplified version by updating the config and re-running evaluation
    
    # Evaluation on target dataset
    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--model", model,
        "--representation", representation,
        "--config", config_path,
        "--test_root", target_dataset,
        "--seed", str(seed),
    ]
    
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    if eval_result.returncode != 0:
        print(f"  Evaluation failed: {eval_result.stderr}")
        return None
    
    print(f"  Evaluation on target dataset completed")
    
    # Load evaluation results
    results_dir = Path(base_config["project"]["results_dir"])
    eval_log_path = (
        results_dir / "logs" / 
        f"eval_{representation + '_' if representation == 'gnn' else ''}{model}_seed{seed}.json"
    )
    
    if eval_log_path.exists():
        with open(eval_log_path) as f:
            eval_log = json.load(f)
        
        # Add cross-dataset metadata
        eval_log["cross_dataset"] = {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
        }
        
        return eval_log
    else:
        print(f"  Warning: Evaluation log not found")
        return None


def run_cross_dataset_sweep(
    model: str,
    representation: str,
    dataset_pairs: List[Tuple[str, str]],
    base_config: Dict,
    temp_config_dir: Path,
    seeds: List[int],
) -> List[Dict]:
    """Run cross-dataset experiments for multiple dataset pairs."""
    
    results = []
    total_experiments = len(dataset_pairs) * len(seeds)
    
    print(f"\n{'='*60}")
    print(f"Cross-dataset sweep for {representation.upper()}-{model}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}")
    
    experiment_count = 0
    
    for source_dataset, target_dataset in dataset_pairs:
        for seed in seeds:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}]")
            
            result = run_single_cross_dataset_experiment(
                model=model,
                representation=representation,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                base_config=base_config,
                temp_config_dir=temp_config_dir,
                seed=seed,
            )
            
            if result is not None:
                results.append(result)
    
    print(f"\nCompleted {len(results)}/{total_experiments} experiments")
    return results


def save_cross_dataset_results(
    results: List[Dict],
    model: str,
    representation: str,
    results_dir: str = "results",
) -> None:
    """Save cross-dataset results to CSV and JSON."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    json_path = results_path / "logs" / f"cross_dataset_{representation}_{model}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved detailed results to {json_path}")
    
    # Create summary CSV
    csv_path = results_path / "tables" / "cross_dataset_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "model", "representation", "seed",
        "source_dataset", "target_dataset",
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
            cross_data = result.get("cross_dataset", {})
            row = {
                "model": f"{representation}_{model}" if representation == "gnn" else model,
                "representation": representation,
                "seed": result.get("seed"),
                "source_dataset": cross_data.get("source_dataset"),
                "target_dataset": cross_data.get("target_dataset"),
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
    
    print(f"Updated cross-dataset results table at {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-dataset generalization experiments"
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
        "--source_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Source dataset paths"
    )
    parser.add_argument(
        "--target_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Target dataset paths"
    )
    parser.add_argument(
        "--dataset_pairs",
        type=str,
        nargs="+",
        default=None,
        help="Dataset pairs in format 'source:target'"
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
    
    # Parse dataset pairs
    dataset_pairs = []
    
    if args.dataset_pairs:
        # Parse source:target format
        for pair in args.dataset_pairs:
            if ":" in pair:
                source, target = pair.split(":", 1)
                dataset_pairs.append((source, target))
    elif args.source_datasets and args.target_datasets:
        # Create all combinations
        for source in args.source_datasets:
            for target in args.target_datasets:
                if source != target:  # Don't test on same dataset
                    dataset_pairs.append((source, target))
    else:
        # Default: use synthetic datasets for testing
        print("No dataset pairs specified, using synthetic dataset fallback")
        source = base_config["data"]["root"]
        target = source  # For testing, use same as source
        dataset_pairs.append((source, target))
    
    # Get seeds
    seeds = args.seeds if args.seeds is not None else base_config["project"].get("seeds_for_stats", [42])
    
    # Create temporary config directory
    temp_config_dir = Path("temp_cross_dataset_configs")
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cross-Dataset Generalization Experiments")
    print(f"Config: {args.config}")
    print(f"Models: {args.models}")
    print(f"Representations: {args.representations}")
    print(f"Dataset pairs: {dataset_pairs}")
    print(f"Seeds: {seeds}")
    
    # Run cross-dataset sweeps
    all_results = []
    
    for model in args.models:
        for representation in args.representations:
            results = run_cross_dataset_sweep(
                model=model,
                representation=representation,
                dataset_pairs=dataset_pairs,
                base_config=base_config,
                temp_config_dir=temp_config_dir,
                seeds=seeds,
            )
            
            if results:
                save_cross_dataset_results(results, model, representation)
                all_results.extend(results)
    
    # Cleanup temporary configs
    import shutil
    if temp_config_dir.exists():
        shutil.rmtree(temp_config_dir)
        print(f"\nCleaned up temporary config directory")
    
    if all_results:
        print(f"\n{'='*60}")
        print("Cross-dataset experiments completed successfully")
        print(f"Total results collected: {len(all_results)}")
        print(f"{'='*60}")
    else:
        print("\nNo successful experiments")
        sys.exit(1)


if __name__ == "__main__":
    main()