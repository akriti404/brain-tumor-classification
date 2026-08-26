"""
CNN explainability analysis using Grad-CAM.

Generates visual explanations for CNN-VQC model predictions by identifying
which image regions contribute most to classification decisions.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import build_dataloaders
from models.hybrid import build_model_from_config
from utils.reproducibility import set_seed, get_device


def load_model_and_data(
    config_path: str,
    model_name: str = "hybrid",
    seed: int = 42,
) -> Tuple[nn.Module, torch.utils.data.DataLoader, List[str], Dict]:
    """Load trained model and data for explainability analysis."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(seed)
    device = get_device(cfg["project"]["device"])
    
    # Build data loaders
    train_loader, val_loader, test_loader, classes, meta = build_dataloaders(cfg)
    n_classes = len(classes)
    
    # Build model
    model = build_model_from_config(cfg, n_classes)
    model.to(device)
    
    # Load checkpoint
    results_dir = Path(cfg["project"]["results_dir"])
    checkpoint_path = results_dir / "checkpoints" / f"{model_name}_seed{seed}.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model.eval()
    return model, test_loader, classes, cfg


def get_target_layers(model: nn.Module) -> List[nn.Module]:
    """Identify target layers for Grad-CAM in the CNN feature extractor."""
    target_layers = []
    
    # Try to find the last convolutional layer in the feature extractor
    if hasattr(model, 'extractor'):
        extractor = model.extractor
        if hasattr(extractor, 'model'):
            # For torchvision models like MobileNetV2, ResNet18
            # Look for the last conv layer before the classifier
            for name, module in reversed(list(extractor.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layers.append(module)
                    print(f"Found target layer: {name}")
                    break
        elif hasattr(extractor, 'features'):
            # For custom CNN architectures
            for name, module in reversed(list(extractor.features.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layers.append(module)
                    print(f"Found target layer: {name}")
                    break
    
    if not target_layers:
        # Fallback: find any Conv2d layer
        for name, module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layers.append(module)
                print(f"Fallback target layer: {name}")
                break
    
    return target_layers


def generate_gradcam_explanation(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
    method: str = "gradcam",
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image."""
    model.eval()
    
    # Get target layers
    target_layers = get_target_layers(model)
    if not target_layers:
        raise ValueError("No suitable convolutional layers found for Grad-CAM")
    
    # Choose CAM method
    if method == "gradcam":
        cam = GradCAM(model=model, target_layers=target_layers)
    elif method == "gradcam++":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unknown CAM method: {method}")
    
    # Prepare input
    input_tensor = image.unsqueeze(0).to(device)
    
    # Generate CAM
    targets = [target_class]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension
    
    return grayscale_cam


def process_single_image(
    model: nn.Module,
    image: torch.Tensor,
    label: int,
    classes: List[str],
    device: torch.device,
    output_dir: Path,
    image_index: int,
    method: str = "gradcam",
) -> Dict:
    """Process a single image and generate explanation."""
    # Get model prediction
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate Grad-CAM
    try:
        heatmap = generate_gradcam_explanation(
            model, image, pred_class, device, method
        )
    except Exception as e:
        print(f"  Failed to generate Grad-CAM: {e}")
        return None
    
    # Convert image to numpy for visualization
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np * std + mean).clip(0, 1)
    
    # Overlay heatmap on image
    visualization = show_cam_on_image(image_np, heatmap, use_rgb=True)
    
    # Save visualization
    class_name = classes[pred_class]
    true_class = classes[label]
    output_path = output_dir / f"exp_{image_index:04d}_pred{class_name}_true{true_class}.png"
    
    # Convert to PIL Image and save
    vis_pil = Image.fromarray(visualization)
    vis_pil.save(output_path)
    
    return {
        "image_index": image_index,
        "true_class": true_class,
        "pred_class": class_name,
        "confidence": confidence,
        "correct": pred_class == label,
        "output_path": str(output_path),
        "method": method,
    }


def run_explainability_analysis(
    config_path: str,
    model_name: str = "hybrid",
    seed: int = 42,
    num_samples: int = 10,
    method: str = "gradcam",
    output_dir: str = "results/explainability/cnn",
) -> List[Dict]:
    """Run explainability analysis on CNN-VQC model."""
    
    print(f"\n{'='*60}")
    print("CNN EXPLAINABILITY ANALYSIS (Grad-CAM)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Samples: {num_samples}")
    print(f"Method: {method}")
    
    # Load model and data
    model, test_loader, classes, cfg = load_model_and_data(
        config_path, model_name, seed
    )
    device = get_device(cfg["project"]["device"])
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images
    results = []
    processed = 0
    
    print(f"\nProcessing images from test set...")
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        if processed >= num_samples:
            break
        
        for i in range(min(len(images), num_samples - processed)):
            image = images[i]
            label = labels[i].item()
            
            result = process_single_image(
                model, image, label, classes, device,
                output_path, processed, method
            )
            
            if result:
                results.append(result)
                print(f"  [{processed+1}/{num_samples}] {result['true_class']} -> {result['pred_class']} ({result['confidence']:.3f})")
            
            processed += 1
    
    # Save results
    results_json = output_path / f"explainability_{model_name}_seed{seed}.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {results_json}")
    print(f"Generated {len(results)} explanations")
    
    # Calculate accuracy on processed samples
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0
    print(f"Accuracy on processed samples: {accuracy:.3f} ({correct}/{len(results)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CNN explainability analysis using Grad-CAM"
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
        "--method",
        type=str,
        choices=["gradcam", "gradcam++"],
        default="gradcam",
        help="Grad-CAM method to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/explainability/cnn",
        help="Output directory for explanations"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_explainability_analysis(
            config_path=args.config,
            model_name=args.model,
            seed=args.seed,
            num_samples=args.num_samples,
            method=args.method,
            output_dir=args.output_dir,
        )
        
        print(f"\n{'='*60}")
        print("CNN explainability analysis completed successfully")
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