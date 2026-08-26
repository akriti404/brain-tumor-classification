"""
Multi-seed validation framework for statistical robustness.

Runs training and evaluation across multiple random seeds to provide
statistical validation of experimental results with mean ± std reporting.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import aggregate_over_seeds


def run_single_seed(
    model: str,
    representation: str,
    seed: int,
    config_path: str,
) -> Dict:
    """Run training and evaluation for a single seed."""
    print(f"\n{'='*60}")
    print(f"Running {representation.upper()}-{model} with seed {seed}")
    print(f"{'='*60}\n")

    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found at {config_path}")
        return None

    # Training
    train_cmd = [
        sys.executable,
        "train.py",
        "--model", model,
        "--representation", representation,
        "--config", config_path,
        "--seed", str(seed),
    ]
    
    print(f"Training command: {' '.join(train_cmd)}")
    try:
        train_result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
    except subprocess.TimeoutExpired:
        print(f"Training timed out for seed {seed}")
        return None
    except Exception as e:
        print(f"Training failed with exception for seed {seed}: {e}")
        return None
    
    if train_result.returncode != 0:
        print(f"Training failed for seed {seed}:")
        print(train_result.stderr)
        return None
    
    print("Training completed successfully")

    # Evaluation
    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--model", model,
        "--representation", representation,
        "--config", config_path,
        "--seed", str(seed),
    ]
    
    print(f"Evaluation command: {' '.join(eval_cmd)}")
    try:
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
    except subprocess.TimeoutExpired:
        print(f"Evaluation timed out for seed {seed}")
        return None
    except Exception as e:
        print(f"Evaluation failed with exception for seed {seed}: {e}")
        return None
    
    if eval_result.returncode != 0:
        print(f"Evaluation failed for seed {seed}:")
        print(eval_result.stderr)
        return None
    
    print("Evaluation completed successfully")
    
    # Load evaluation log
    results_dir = Path("results")
    eval_log_path = (
        results_dir / "logs" / 
        f"eval_{representation + '_' if representation == 'gnn' else ''}{model}_seed{seed}.json"
    )
    
    if eval_log_path.exists():
        try:
            with open(eval_log_path) as f:
                eval_log = json.load(f)
            return eval_log
        except json.JSONDecodeError as e:
            print(f"Error loading evaluation log: {e}")
            return None
    else:
        print(f"Warning: Evaluation log not found at {eval_log_path}")
        return None


def aggregate_results(
    model: str,
    representation: str,
    seeds: List[int],
    config_path: str,
) -> Dict:
    """Run experiments across multiple seeds and aggregate results."""
    print(f"\n{'='*60}")
    print(f"Multi-seed validation for {representation.upper()}-{model}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}\n")

    individual_results = []
    
    for seed in seeds:
        result = run_single_seed(model, representation, seed, config_path)
        if result is not None:
            individual_results.append(result)
        else:
            print(f"Failed to complete experiment for seed {seed}")
    
    if not individual_results:
        print("No successful experiments to aggregate")
        return None
    
    print(f"\nSuccessfully completed {len(individual_results)}/{len(seeds)} experiments")
    
    # Aggregate results
    aggregated = aggregate_over_seeds(individual_results)
    
    # Add metadata
    aggregated["model"] = f"{representation}_{model}" if representation == "gnn" else model
    aggregated["representation"] = representation
    aggregated["seeds"] = seeds
    aggregated["n_successful_runs"] = len(individual_results)
    
    return aggregated


def save_aggregated_results(
    aggregated: Dict,
    results_dir: str = "results",
) -> None:
    """Save aggregated results to JSON and update CSV table."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    model_name = aggregated["model"]
    json_path = results_path / "logs" / f"multi_seed_{model_name}.json"
    
    with open(json_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Saved aggregated results to {json_path}")
    
    # Update CSV table with summary statistics
    csv_path = results_path / "tables" / "multi_seed_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract summary metrics for CSV
    summary_row = {
        "model": model_name,
        "representation": aggregated["representation"],
        "seeds": str(aggregated["seeds"]),
        "n_successful_runs": aggregated["n_successful_runs"],
    }
    
    # Add mean ± std for key metrics
    for metric_name, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            summary_row[f"{metric_name}_mean"] = stats["mean"]
            summary_row[f"{metric_name}_std"] = stats["std"]
            summary_row[f"{metric_name}_n"] = stats["n_seeds"]
    
    # Write to CSV
    write_header = not csv_path.exists()
    
    with open(csv_path, "a", newline="") as f:
        import csv
        fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        writer.writerow(summary_row)
    
    print(f"Updated multi-seed results table at {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed validation experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hybrid", "simple_cnn", "resnet18", "mobilenet_v2", "classical_proposed"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["cnn", "gnn"],
        default="cnn",
        help="Representation type (cnn or gnn)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Specific seeds to run (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Get seeds from config or command line
    seeds = args.seeds if args.seeds is not None else cfg["project"].get("seeds_for_stats", [42])
    
    print(f"Configuration loaded from {args.config}")
    print(f"Model: {args.model}")
    print(f"Representation: {args.representation}")
    print(f"Seeds: {seeds}")
    
    # Run multi-seed validation
    aggregated = aggregate_results(
        model=args.model,
        representation=args.representation,
        seeds=seeds,
        config_path=args.config,
    )
    
    if aggregated is not None:
        save_aggregated_results(aggregated)
        print("\n" + "="*60)
        print("Multi-seed validation completed successfully")
        print("="*60)
        print("\nSummary Statistics:")
        for metric_name, stats in aggregated.items():
            if isinstance(stats, dict) and "mean" in stats:
                print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n_seeds']})")
    else:
        print("\nMulti-seed validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()