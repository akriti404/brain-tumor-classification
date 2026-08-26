"""
Noise model experiments for NISQ device simulation.

Systematically evaluates the impact of different noise types and probabilities
on quantum circuit performance for both CNN-VQC and GNN-VQC branches.
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_noise_config(
    base_config: Dict,
    noise_type: str,
    noise_prob: float,
) -> Dict:
    """Create a modified config for a specific noise setting."""
    config = copy.deepcopy(base_config)
    config["quantum"]["noise_type"] = noise_type
    config["quantum"]["noise_prob"] = noise_prob
    return config


def save_temp_config(config: Dict, temp_path: Path) -> str:
    """Save temporary config file."""
    with open(temp_path, "w") as f:
        yaml.dump(config, f)
    return str(temp_path)


def run_single_noise_experiment(
    model: str,
    representation: str,
    noise_type: str,
    noise_prob: float,
    base_config: Dict,
    temp_config_dir: Path,
    seed: int = 42,
) -> Dict:
    """Run a single noise experiment."""
    
    # Create noise-specific config
    noise_config = create_noise_config(base_config, noise_type, noise_prob)
    
    # Save temporary config
    temp_config_path = temp_config_dir / f"temp_noise_{noise_type}_p{noise_prob}.yaml"
    config_path = save_temp_config(noise_config, temp_config_path)
    
    print(f"\nRunning noise experiment: {representation}-{model}")
    print(f"  Noise type: {noise_type}, Probability: {noise_prob}")
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
            
            # Add noise metadata
            eval_log["noise"] = {
                "noise_type": noise_type,
                "noise_prob": noise_prob,
            }
            
            print(f"  Completed successfully")
            return eval_log
        except json.JSONDecodeError as e:
            print(f"  Error loading evaluation log: {e}")
            return None
    else:
        print(f"  Warning: Evaluation log not found")
        return None


def run_noise_sweep(
    model: str,
    representation: str,
    noise_types: List[str],
    noise_probabilities: List[float],
    base_config: Dict,
    temp_config_dir: Path,
    seeds: List[int],
) -> List[Dict]:
    """Run a full noise sweep for a model-representation pair."""
    
    results = []
    total_experiments = (
        len(noise_types) * len(noise_probabilities) * len(seeds)
    )
    
    print(f"\n{'='*60}")
    print(f"Noise sweep for {representation.upper()}-{model}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}")
    
    experiment_count = 0
    
    for noise_type in noise_types:
        for noise_prob in noise_probabilities:
            for seed in seeds:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}]")
                
                result = run_single_noise_experiment(
                    model=model,
                    representation=representation,
                    noise_type=noise_type,
                    noise_prob=noise_prob,
                    base_config=base_config,
                    temp_config_dir=temp_config_dir,
                    seed=seed,
                )
                
                if result is not None:
                    results.append(result)
    
    print(f"\nCompleted {len(results)}/{total_experiments} experiments")
    return results


def save_noise_results(
    results: List[Dict],
    model: str,
    representation: str,
    results_dir: str = "results",
) -> None:
    """Save noise results to CSV and JSON."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    json_path = results_path / "logs" / f"noise_{representation}_{model}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved detailed results to {json_path}")
    
    # Create summary CSV
    csv_path = results_path / "tables" / "noise_experiments.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "model", "representation", "seed",
        "noise_type", "noise_prob",
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
            noise = result.get("noise", {})
            row = {
                "model": f"{representation}_{model}" if representation == "gnn" else model,
                "representation": representation,
                "seed": result.get("seed"),
                "noise_type": noise.get("noise_type"),
                "noise_prob": noise.get("noise_prob"),
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
    
    print(f"Updated noise results table at {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run noise model experiments"
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
        "--noise_types",
        type=str,
        nargs="+",
        choices=["ideal", "bit_flip", "phase_flip", "depolarizing"],
        default=["ideal", "bit_flip", "phase_flip", "depolarizing"],
        help="Noise types to evaluate"
    )
    parser.add_argument(
        "--noise_probs",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1],
        help="Noise probabilities to evaluate"
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
    
    # Get seeds
    seeds = args.seeds if args.seeds is not None else base_config["project"].get("seeds_for_stats", [42])
    
    # Create temporary config directory
    temp_config_dir = Path("temp_noise_configs")
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Noise Model Experiments")
    print(f"Config: {args.config}")
    print(f"Models: {args.models}")
    print(f"Representations: {args.representations}")
    print(f"Noise types: {args.noise_types}")
    print(f"Noise probabilities: {args.noise_probs}")
    print(f"Seeds: {seeds}")
    
    # Run noise sweeps
    all_results = []
    
    for model in args.models:
        for representation in args.representations:
            results = run_noise_sweep(
                model=model,
                representation=representation,
                noise_types=args.noise_types,
                noise_probabilities=args.noise_probs,
                base_config=base_config,
                temp_config_dir=temp_config_dir,
                seeds=seeds,
            )
            
            if results:
                save_noise_results(results, model, representation)
                all_results.extend(results)
    
    # Cleanup temporary configs
    import shutil
    if temp_config_dir.exists():
        shutil.rmtree(temp_config_dir)
        print(f"\nCleaned up temporary config directory")
    
    if all_results:
        print(f"\n{'='*60}")
        print("Noise experiments completed successfully")
        print(f"Total results collected: {len(all_results)}")
        print(f"{'='*60}")
    else:
        print("\nNo successful experiments")
        sys.exit(1)


if __name__ == "__main__":
    main()