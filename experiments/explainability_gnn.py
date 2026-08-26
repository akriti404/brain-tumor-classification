"""
GNN explainability analysis using GNNExplainer.

Generates visual explanations for GNN-VQC model predictions by identifying
important subgraph structures and node features that contribute to decisions.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.graph import build_graph_dataloaders
from models.gnn_hybrid import build_gnn_model_from_config
from utils.reproducibility import set_seed, get_device


def load_gnn_model_and_data(
    config_path: str,
    model_name: str = "hybrid",
    seed: int = 42,
):
    """Load trained GNN model and data for explainability analysis."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(seed)
    device = get_device(cfg["project"]["device"])
    
    # Build data loaders
    train_loader, val_loader, test_loader, classes, meta = build_graph_dataloaders(cfg)
    n_classes = len(classes)
    
    # Build model
    model = build_gnn_model_from_config(cfg, n_classes)
    model.to(device)
    
    # Load checkpoint
    results_dir = Path(cfg["project"]["results_dir"])
    checkpoint_path = results_dir / "checkpoints" / f"gnn_{model_name}_seed{seed}.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model.eval()
    return model, test_loader, classes, cfg, device


def process_single_graph(
    model,
    graph_data: Data,
    true_label: int,
    classes: List[str],
    device: torch.device,
    output_dir: Path,
    graph_index: int,
) -> Dict:
    """Process a single graph and generate explanation."""
    try:
        # Try new PyG 2.3+ API
        from torch_geometric.explain import Explainer, GNNExplainer
        use_new_api = True
    except ImportError:
        try:
            # Fallback to old API
            from torch_geometric.nn import GNNExplainer
            use_new_api = False
        except ImportError:
            print("GNNExplainer not available. Install torch-geometric >= 2.3")
            return None
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        graph_data = graph_data.to(device)
        logits = model(graph_data)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Setup explainer based on API version
    if use_new_api:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100, lr=0.01),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        
        # Generate explanation
        try:
            explanation = explainer(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                target=torch.tensor([pred_class]),
                index=0,
            )
            node_mask = explanation.node_mask
            edge_mask = explanation.edge_mask
        except Exception as e:
            print(f"  Failed to generate explanation: {e}")
            return None
    else:
        # Use old API
        explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=True)
        
        try:
            node_mask, edge_mask = explainer.explain_graph(
                graph_data.x, graph_data.edge_index
            )
        except Exception as e:
            print(f"  Failed to generate explanation: {e}")
            return None
    
    # Get most important nodes
    if node_mask is not None and node_mask.dim() > 1:
        # For attribute masks, aggregate across features
        node_importance = node_mask.mean(dim=1).cpu().numpy()
    else:
        node_importance = node_mask.cpu().numpy() if node_mask is not None else np.ones(graph_data.num_nodes)
    
    # Get most important edges
    edge_importance = edge_mask.cpu().numpy() if edge_mask is not None else np.ones(graph_data.edge_index.shape[1])
    
    # Save explanation data
    class_name = classes[pred_class]
    true_class = classes[true_label]
    
    explanation_data = {
        "graph_index": graph_index,
        "num_nodes": graph_data.num_nodes,
        "num_edges": graph_data.edge_index.shape[1],
        "true_class": true_class,
        "pred_class": class_name,
        "confidence": confidence,
        "correct": pred_class == true_label,
        "node_importance": node_importance.tolist(),
        "edge_importance": edge_importance.tolist(),
        "node_features": graph_data.x.cpu().numpy().tolist(),
        "edge_index": graph_data.edge_index.cpu().numpy().tolist(),
    }
    
    # Save individual explanation
    exp_path = output_dir / f"exp_{graph_index:04d}_pred{class_name}_true{true_class}.json"
    with open(exp_path, "w") as f:
        json.dump(explanation_data, f, indent=2)
    
    return explanation_data


def run_gnn_explainability_analysis(
    config_path: str,
    model_name: str = "hybrid",
    seed: int = 42,
    num_samples: int = 10,
    output_dir: str = "results/explainability/gnn",
) -> List[Dict]:
    """Run explainability analysis on GNN-VQC model."""
    
    print(f"\n{'='*60}")
    print("GNN EXPLAINABILITY ANALYSIS (GNNExplainer)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Samples: {num_samples}")
    
    # Load model and data
    model, test_loader, classes, cfg, device = load_gnn_model_and_data(
        config_path, model_name, seed
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process graphs
    results = []
    processed = 0
    
    print(f"\nProcessing graphs from test set...")
    
    for batch_idx, batch in enumerate(test_loader):
        if processed >= num_samples:
            break
        
        # Unbatch individual graphs
        for i in range(min(len(batch), num_samples - processed)):
            # Get individual graph from batch
            graph_data = batch[i].clone()
            true_label = graph_data.y.item()
            
            result = process_single_graph(
                model, graph_data, true_label, classes, device,
                output_path, processed
            )
            
            if result:
                results.append(result)
                print(f"  [{processed+1}/{num_samples}] {result['true_class']} -> {result['pred_class']} ({result['confidence']:.3f})")
            
            processed += 1
    
    # Save aggregated results
    results_json = output_path / f"explainability_{model_name}_seed{seed}.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {results_json}")
    print(f"Generated {len(results)} explanations")
    
    # Calculate accuracy on processed samples
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0
    print(f"Accuracy on processed samples: {accuracy:.3f} ({correct}/{len(results)})")
    
    # Analyze feature importance
    if results:
        analyze_feature_importance(results, output_path)
    
    return results


def analyze_feature_importance(results: List[Dict], output_dir: Path):
    """Analyze and summarize feature importance across explanations."""
    print(f"\nAnalyzing feature importance patterns...")
    
    # Collect node importance statistics
    all_node_importance = []
    for result in results:
        all_node_importance.extend(result["node_importance"])
    
    all_node_importance = np.array(all_node_importance)
    
    importance_stats = {
        "mean": float(np.mean(all_node_importance)),
        "std": float(np.std(all_node_importance)),
        "min": float(np.min(all_node_importance)),
        "max": float(np.max(all_node_importance)),
        "median": float(np.median(all_node_importance)),
    }
    
    print(f"Node importance statistics:")
    for stat, value in importance_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    # Save analysis
    analysis_path = output_dir / "feature_importance_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "importance_stats": importance_stats,
            "num_graphs_analyzed": len(results),
            "total_nodes": len(all_node_importance),
        }, f, indent=2)
    
    print(f"Saved feature importance analysis to {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GNN explainability analysis using GNNExplainer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        help="Model to explain"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of test samples to explain"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/explainability/gnn",
        help="Output directory for explanations"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_gnn_explainability_analysis(
            config_path=args.config,
            model_name=args.model,
            seed=args.seed,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )
        
        print(f"\n{'='*60}")
        print("GNN explainability analysis completed successfully")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during explainability analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())