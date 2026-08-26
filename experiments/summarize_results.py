"""
Experiment result summary utility for review panel preparation.

Generates comprehensive summaries of all experimental results for
easy presentation and documentation.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_all_results(results_dir: str = "results") -> Dict:
    """Load all available experimental results."""
    results_path = Path(results_dir)
    
    summary = {
        "results_dir": str(results_path),
        "experiments_completed": [],
        "experiments_available": [],
        "summaries": {}
    }
    
    # Check for multi-seed results
    multi_seed_dir = results_path / "logs"
    for json_file in multi_seed_dir.glob("multi_seed_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        model_name = data.get("model", json_file.stem)
        summary["experiments_completed"].append(f"multi_seed_{model_name}")
        summary["summaries"][f"multi_seed_{model_name}"] = data
    
    # Check for ablation results
    for json_file in multi_seed_dir.glob("ablation_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        model_name = json_file.stem.replace("ablation_", "")
        summary["experiments_completed"].append(f"ablation_{model_name}")
        summary["summaries"][f"ablation_{model_name}"] = {
            "num_experiments": len(data),
            "configurations": list(set(
                f"q{r.get('ablation', {}).get('n_qubits')}_l{r.get('ablation', {}).get('n_layers')}_ru{r.get('ablation', {}).get('data_reuploading')}"
                for r in data
            ))
        }
    
    # Check for noise results
    for json_file in multi_seed_dir.glob("noise_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        model_name = json_file.stem.replace("noise_", "")
        summary["experiments_completed"].append(f"noise_{model_name}")
        summary["summaries"][f"noise_{model_name}"] = {
            "num_experiments": len(data),
            "noise_types": list(set(
                r.get('noise', {}).get('noise_type') for r in data
            ))
        }
    
    # Check for cross-dataset results
    for json_file in multi_seed_dir.glob("cross_dataset_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        model_name = json_file.stem.replace("cross_dataset_", "")
        summary["experiments_completed"].append(f"cross_dataset_{model_name}")
        summary["summaries"][f"cross_dataset_{model_name}"] = {
            "num_experiments": len(data),
            "dataset_pairs": list(set(
                f"{r.get('cross_dataset', {}).get('source_dataset')} -> {r.get('cross_dataset', {}).get('target_dataset')}"
                for r in data
            ))
        }
    
    # Check for CSV tables
    tables_dir = results_path / "tables"
    if tables_dir.exists():
        for csv_file in tables_dir.glob("*.csv"):
            summary["experiments_available"].append(csv_file.stem)
            try:
                df = pd.read_csv(csv_file)
                summary["summaries"][f"table_{csv_file.stem}"] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "file_size_mb": csv_file.stat().st_size / (1024 * 1024)
                }
            except Exception as e:
                summary["summaries"][f"table_{csv_file.stem}"] = {"error": str(e)}
    
    # Check for figures
    figures_dir = results_path / "figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        summary["figures_available"] = len(figures)
        summary["figure_files"] = [f.name for f in figures]
    
    # Check for explainability results
    explainability_dir = results_path / "explainability"
    if explainability_dir.exists():
        summary["explainability_available"] = True
        cnn_dir = explainability_dir / "cnn"
        gnn_dir = explainability_dir / "gnn"
        
        summary["explainability"] = {}
        if cnn_dir.exists():
            cnn_files = list(cnn_dir.glob("*"))
            summary["explainability"]["cnn"] = {
                "num_files": len(cnn_files),
                "files": [f.name for f in cnn_files]
            }
        if gnn_dir.exists():
            gnn_files = list(gnn_dir.glob("*"))
            summary["explainability"]["gnn"] = {
                "num_files": len(gnn_files),
                "files": [f.name for f in gnn_files]
            }
    
    return summary


def generate_performance_summary(results_dir: str = "results") -> pd.DataFrame:
    """Generate a performance summary table from all available results."""
    results_path = Path(results_dir)
    tables_dir = results_path / "tables"
    
    performance_data = []
    
    # Load baseline results
    baseline_csv = tables_dir / "experiment_results.csv"
    if baseline_csv.exists():
        df = pd.read_csv(baseline_csv)
        for _, row in df.iterrows():
            performance_data.append({
                "experiment": "baseline",
                "model": row["model"],
                "representation": row.get("dataset_split_method", "unknown"),
                "accuracy": row["accuracy"],
                "f1_macro": row.get("f1_macro"),
                "total_params": row.get("total_params"),
                "quantum_params": row.get("quantum_parameters"),
            })
    
    # Load multi-seed results
    multi_seed_csv = tables_dir / "multi_seed_results.csv"
    if multi_seed_csv.exists():
        df = pd.read_csv(multi_seed_csv)
        for _, row in df.iterrows():
            performance_data.append({
                "experiment": "multi_seed",
                "model": row["model"],
                "representation": row.get("representation", "unknown"),
                "accuracy": row.get("accuracy_mean"),
                "f1_macro": row.get("f1_macro_mean"),
                "total_params": row.get("total_params"),
                "quantum_params": row.get("quantum_parameters"),
            })
    
    # Load ablation results
    ablation_csv = tables_dir / "resource_ablations.csv"
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)
        for _, row in df.iterrows():
            performance_data.append({
                "experiment": f"ablation_q{row['n_qubits']}_l{row['n_layers']}",
                "model": row["model"],
                "representation": row["representation"],
                "accuracy": row["accuracy"],
                "f1_macro": row.get("f1_macro"),
                "total_params": row.get("total_params"),
                "quantum_params": row.get("quantum_parameters"),
            })
    
    # Load noise results
    noise_csv = tables_dir / "noise_experiments.csv"
    if noise_csv.exists():
        df = pd.read_csv(noise_csv)
        for _, row in df.iterrows():
            performance_data.append({
                "experiment": f"noise_{row['noise_type']}_p{row['noise_prob']}",
                "model": row["model"],
                "representation": row["representation"],
                "accuracy": row["accuracy"],
                "f1_macro": row.get("f1_macro"),
                "total_params": row.get("total_params"),
                "quantum_params": row.get("quantum_parameters"),
            })
    
    return pd.DataFrame(performance_data)


def generate_markdown_report(summary: Dict, output_path: str) -> None:
    """Generate a markdown report for review panel."""
    
    md_content = f"""# Experimental Results Summary

**Generated on**: {summary.get('results_dir', 'unknown')}
**Experiments Completed**: {len(summary.get('experiments_completed', []))}
**Experiments Available**: {len(summary.get('experiments_available', []))}
**Figures Available**: {summary.get('figures_available', 0)}

## Completed Experiments

"""
    
    for exp in summary.get('experiments_completed', []):
        md_content += f"- {exp}\n"
    
    md_content += "\n## Available Data Tables\n\n"
    
    for table in summary.get('experiments_available', []):
        table_info = summary['summaries'].get(f'table_{table}', {})
        md_content += f"### {table}\n"
        if 'rows' in table_info:
            md_content += f"- Rows: {table_info['rows']}\n"
            md_content += f"- Columns: {', '.join(table_info['columns'][:5])}...\n"
            md_content += f"- Size: {table_info.get('file_size_mb', 0):.2f} MB\n"
        md_content += "\n"
    
    md_content += "\n## Figures\n\n"
    
    if summary.get('figures_available', 0) > 0:
        for fig in summary.get('figure_files', []):
            md_content += f"- `{fig}`\n"
    else:
        md_content += "No figures generated yet.\n"
    
    md_content += "\n## Explainability Results\n\n"
    
    if summary.get('explainability_available', False):
        explain_data = summary.get('explainability', {})
        if 'cnn' in explain_data:
            md_content += f"### CNN Explainability\n"
            md_content += f"- Files: {explain_data['cnn']['num_files']}\n"
        if 'gnn' in explain_data:
            md_content += f"### GNN Explainability\n"
            md_content += f"- Files: {explain_data['gnn']['num_files']}\n"
    else:
        md_content += "No explainability results available.\n"
    
    md_content += "\n## Key Findings Summary\n\n"
    md_content += "*This section will be populated after statistical analysis is completed.*\n"
    
    with open(output_path, "w") as f:
        f.write(md_content)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experimental results for review panel"
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
        default="results/summary",
        help="Output directory for summary files"
    )
    
    args = parser.parse_args()
    
    print("Generating experimental results summary...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load all results
    summary = load_all_results(args.results_dir)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    json_path = output_path / "results_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to {json_path}")
    
    # Generate performance summary table
    performance_df = generate_performance_summary(args.results_dir)
    if not performance_df.empty:
        csv_path = output_path / "performance_summary.csv"
        performance_df.to_csv(csv_path, index=False)
        print(f"Saved performance summary to {csv_path}")
    else:
        print("No performance data available for summary table")
    
    # Generate markdown report
    md_path = output_path / "results_summary.md"
    generate_markdown_report(summary, md_path)
    print(f"Saved markdown report to {md_path}")
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("EXPERIMENTAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Completed experiments: {len(summary['experiments_completed'])}")
    print(f"Available tables: {len(summary['experiments_available'])}")
    print(f"Available figures: {summary.get('figures_available', 0)}")
    print(f"Explainability available: {summary.get('explainability_available', False)}")
    
    if summary['experiments_completed']:
        print(f"\nCompleted experiments:")
        for exp in summary['experiments_completed']:
            print(f"  - {exp}")
    
    print(f"\n{'='*60}")
    print("Summary generation completed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()