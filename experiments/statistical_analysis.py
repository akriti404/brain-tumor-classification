"""
Statistical analysis framework for experimental validation.

Performs hypothesis testing, confidence intervals, and effect size measurements
to provide rigorous statistical validation of experimental claims.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    compare_groups,
    compute_confidence_interval,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


def load_multi_seed_results(
    results_dir: str = "results",
) -> Dict[str, List[Dict]]:
    """Load multi-seed results for different models and representations."""
    results_path = Path(results_dir)
    multi_seed_dir = results_path / "logs"
    
    results = {}
    
    # Look for multi-seed result files
    for json_file in multi_seed_dir.glob("multi_seed_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        model_name = data.get("model", json_file.stem)
        results[model_name] = data
    
    return results


def load_ablation_results(
    results_dir: str = "results",
) -> Dict[str, List[Dict]]:
    """Load resource ablation results."""
    results_path = Path(results_dir)
    ablation_dir = results_path / "logs"
    
    results = {}
    
    for json_file in ablation_dir.glob("ablation_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        model_name = json_file.stem.replace("ablation_", "")
        results[model_name] = data
    
    return results


def load_noise_results(
    results_dir: str = "results",
) -> Dict[str, List[Dict]]:
    """Load noise experiment results."""
    results_path = Path(results_dir)
    noise_dir = results_path / "logs"
    
    results = {}
    
    for json_file in noise_dir.glob("noise_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        model_name = json_file.stem.replace("noise_", "")
        results[model_name] = data
    
    return results


def compare_cnn_vs_gnn(
    cnn_results: List[Dict],
    gnn_results: List[Dict],
    metric: str = "accuracy",
    confidence: float = 0.95,
) -> Dict:
    """Perform statistical comparison between CNN and GNN representations."""
    
    print(f"\n{'='*60}")
    print(f"CNN vs GNN Statistical Comparison (metric: {metric})")
    print(f"{'='*60}")
    
    comparison = compare_groups(
        group1_metrics=cnn_results,
        group2_metrics=gnn_results,
        metric_key=metric,
        confidence=confidence,
    )
    
    # Print summary
    print(f"\nCNN Group:")
    print(f"  Mean: {comparison['group1']['mean']:.4f}")
    print(f"  Std: {comparison['group1']['std']:.4f}")
    print(f"  CI ({confidence*100}%): [{comparison['group1']['ci_lower']:.4f}, {comparison['group1']['ci_upper']:.4f}]")
    
    print(f"\nGNN Group:")
    print(f"  Mean: {comparison['group2']['mean']:.4f}")
    print(f"  Std: {comparison['group2']['std']:.4f}")
    print(f"  CI ({confidence*100}%): [{comparison['group2']['ci_lower']:.4f}, {comparison['group2']['ci_upper']:.4f}]")
    
    if 'independent_t_test' in comparison:
        print(f"\nIndependent t-test:")
        print(f"  Statistic: {comparison['independent_t_test']['statistic']:.4f}")
        print(f"  p-value: {comparison['independent_t_test']['p_value']:.4f}")
        print(f"  Significant (α=0.05): {comparison['independent_t_test']['significant']}")
    
    if 'paired_t_test' in comparison:
        print(f"\nPaired t-test:")
        print(f"  Statistic: {comparison['paired_t_test']['statistic']:.4f}")
        print(f"  p-value: {comparison['paired_t_test']['p_value']:.4f}")
        print(f"  Significant (α=0.05): {comparison['paired_t_test']['significant']}")
    
    if 'effect_size' in comparison:
        print(f"\nEffect Size (Cohen's d):")
        print(f"  Value: {comparison['effect_size']['value']:.4f}")
        print(f"  Interpretation: {comparison['effect_size']['interpretation']}")
    
    return comparison


def analyze_ablation_trends(
    ablation_results: List[Dict],
    metric: str = "accuracy",
) -> Dict:
    """Analyze trends in resource ablation experiments."""
    
    print(f"\n{'='*60}")
    print(f"Resource Ablation Trend Analysis (metric: {metric})")
    print(f"{'='*60}")
    
    # Group by ablation parameters
    qubit_effects = {}
    layer_effects = {}
    reuploading_effects = {}
    
    for result in ablation_results:
        ablation = result.get("ablation", {})
        n_qubits = ablation.get("n_qubits")
        n_layers = ablation.get("n_layers")
        reuploading = ablation.get("data_reuploading")
        value = result.get(metric)
        
        if value is not None:
            # Qubit effects
            if n_qubits not in qubit_effects:
                qubit_effects[n_qubits] = []
            qubit_effects[n_qubits].append(value)
            
            # Layer effects
            if n_layers not in layer_effects:
                layer_effects[n_layers] = []
            layer_effects[n_layers].append(value)
            
            # Re-uploading effects
            if reuploading not in reuploading_effects:
                reuploading_effects[reuploading] = []
            reuploading_effects[reuploading].append(value)
    
    # Compute statistics for each group
    analysis = {
        "qubit_effects": {},
        "layer_effects": {},
        "reuploading_effects": {},
    }
    
    for n_qubits, values in qubit_effects.items():
        analysis["qubit_effects"][n_qubits] = compute_confidence_interval(values)
    
    for n_layers, values in layer_effects.items():
        analysis["layer_effects"][n_layers] = compute_confidence_interval(values)
    
    for reuploading, values in reuploading_effects.items():
        analysis["reuploading_effects"][reuploading] = compute_confidence_interval(values)
    
    # Print summary
    print(f"\nQubit Effects:")
    for n_qubits, stats in sorted(analysis["qubit_effects"].items()):
        print(f"  {n_qubits} qubits: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nLayer Effects:")
    for n_layers, stats in sorted(analysis["layer_effects"].items()):
        print(f"  {n_layers} layers: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nRe-uploading Effects:")
    for reuploading, stats in analysis["reuploading_effects"].items():
        print(f"  {reuploading}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return analysis


def analyze_noise_effects(
    noise_results: List[Dict],
    metric: str = "accuracy",
) -> Dict:
    """Analyze the impact of different noise types and probabilities."""
    
    print(f"\n{'='*60}")
    print(f"Noise Model Impact Analysis (metric: {metric})")
    print(f"{'='*60}")
    
    # Group by noise type and probability
    noise_groups = {}
    
    for result in noise_results:
        noise = result.get("noise", {})
        noise_type = noise.get("noise_type")
        noise_prob = noise.get("noise_prob")
        value = result.get(metric)
        
        if value is not None:
            key = (noise_type, noise_prob)
            if key not in noise_groups:
                noise_groups[key] = []
            noise_groups[key].append(value)
    
    # Compute statistics
    analysis = {}
    
    for (noise_type, noise_prob), values in noise_groups.items():
        key = f"{noise_type}_p{noise_prob}"
        analysis[key] = compute_confidence_interval(values)
    
    # Print summary
    print(f"\nNoise Effects:")
    for key, stats in sorted(analysis.items()):
        print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return analysis


def generate_statistical_report(
    config: Dict,
    results_dir: str = "results",
    output_dir: str = "results/statistical_analysis",
) -> Dict:
    """Generate comprehensive statistical analysis report."""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = {
        "config_summary": {
            "project": config.get("project", {}).get("name", "unknown"),
            "seeds_for_stats": config.get("project", {}).get("seeds_for_stats", []),
        },
        "analyses": {},
    }
    
    # Load results
    multi_seed_results = load_multi_seed_results(results_dir)
    ablation_results = load_ablation_results(results_dir)
    noise_results = load_noise_results(results_dir)
    
    # CNN vs GNN comparison
    # Note: This requires individual seed results, not aggregated multi-seed results
    # For now, we'll skip this if individual seed results aren't available
    print("\nNote: CNN vs GNN comparison requires individual seed results.")
    print("This analysis will be performed when individual seed results are available.")
    
    # Placeholder for future implementation
    report["analyses"]["cnn_vs_gnn_comparison"] = {
        "status": "requires_individual_seed_results",
        "note": "Load individual seed JSON files for paired statistical tests"
    }
    
    # Ablation analysis
    if ablation_results:
        for model_name, results in ablation_results.items():
            if results:
                ablation_analysis = analyze_ablation_trends(results)
                report["analyses"][f"ablation_{model_name}"] = ablation_analysis
    
    # Noise analysis
    if noise_results:
        for model_name, results in noise_results.items():
            if results:
                noise_analysis = analyze_noise_effects(results)
                report["analyses"][f"noise_{model_name}"] = noise_analysis
    
    # Save report
    report_path = output_path / "statistical_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nStatistical analysis report saved to {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on experimental results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing experimental results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/statistical_analysis",
        help="Output directory for statistical analysis"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Primary metric for analysis"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"Statistical Analysis")
    print(f"Config: {args.config}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Primary metric: {args.metric}")
    
    try:
        report = generate_statistical_report(
            config=config,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
        )
        
        print(f"\n{'='*60}")
        print("Statistical analysis completed successfully")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during statistical analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())