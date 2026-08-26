"""
Smoke test for Member 3 experimental framework.

Runs minimal versions of experiments to verify the framework works correctly.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def test_imports():
    """Test that all required dependencies can be imported."""
    print("Testing imports...")
    
    try:
        import yaml
        print("[OK] yaml")
    except ImportError:
        print("[FAIL] yaml")
        return False
    
    try:
        import pandas
        print("[OK] pandas")
    except ImportError:
        print("[FAIL] pandas")
        return False
    
    try:
        import numpy
        print("[OK] numpy")
    except ImportError:
        print("[FAIL] numpy")
        return False
    
    try:
        from scipy import stats
        print("[OK] scipy.stats")
    except ImportError:
        print("[FAIL] scipy.stats")
        return False
    
    try:
        import torch
        print("[OK] torch")
    except ImportError:
        print("[WARN] torch (not installed - install with: pip install torch torchvision)")
        # Don't fail for torch, as it's a runtime dependency
    
    try:
        import pennylane
        print("[OK] pennylane")
    except ImportError:
        print("[FAIL] pennylane")
        return False
    
    try:
        from pytorch_grad_cam import GradCAM
        print("[OK] pytorch_grad_cam")
    except ImportError:
        print("[FAIL] pytorch_grad_cam")
        return False
    
    try:
        import torch_geometric
        print("[OK] torch_geometric")
    except ImportError:
        print("[FAIL] torch_geometric")
        return False
    
    return True


def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")
    
    config_paths = [
        "configs/config.yaml",
        "configs/experimental_examples.yaml"
    ]
    
    for config_path in config_paths:
        if Path(config_path).exists():
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                print(f"[OK] {config_path}")
            except Exception as e:
                print(f"[FAIL] {config_path}: {e}")
                return False
        else:
            print(f"[WARN] {config_path} not found (skipping)")
    
    return True


def test_script_syntax():
    """Test that all experiment scripts have valid syntax."""
    print("\nTesting script syntax...")
    
    scripts = [
        "experiments/multi_seed_runner.py",
        "experiments/resource_ablations.py",
        "experiments/noise_experiments.py",
        "experiments/explainability_cnn.py",
        "experiments/explainability_gnn.py",
        "experiments/cross_dataset.py",
        "experiments/statistical_analysis.py",
        "experiments/run_member3_experiments.py",
        "experiments/summarize_results.py",
        "experiments/progress_tracker.py",
        "experiments/test_experiments.py",  # Test self
    ]
    
    for script in scripts:
        if Path(script).exists():
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    compile(f.read(), script, 'exec')
                print(f"✓ {script}")
            except SyntaxError as e:
                print(f"✗ {script}: {e}")
                return False
            except UnicodeDecodeError as e:
                print(f"✗ {script}: Encoding error - {e}")
                return False
        else:
            print(f"⚠ {script} not found (skipping)")
    
    return True


def test_directory_structure():
    """Test that required directory structure exists."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "experiments",
        "data",
        "models",
        "utils",
        "visualization",
        "configs",
        "results",
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ not found")
            return False
    
    return True


def test_progress_tracker():
    """Test the progress tracker utility."""
    print("\nTesting progress tracker...")
    
    try:
        from experiments.progress_tracker import ExperimentTracker
        
        # Create a test tracker
        tracker = ExperimentTracker("test_experiment", "results")
        tracker.set_total_tasks(3)
        tracker.start_task("task1")
        tracker.complete_task("task1")
        tracker.save_checkpoint("test_checkpoint", {"test": "data"})
        tracker.mark_complete()
        
        # Clean up
        test_file = Path("results/progress/test_experiment_progress.json")
        if test_file.exists():
            test_file.unlink()
        
        print("✓ Progress tracker")
        return True
    except Exception as e:
        print(f"✗ Progress tracker: {e}")
        return False


def run_quick_train_test():
    """Run a very quick training test to verify the pipeline works."""
    print("\nRunning quick training test...")
    print("This will train for 1 epoch with minimal settings")
    
    cmd = [
        sys.executable,
        "train.py",
        "--model", "simple_cnn",
        "--config", "configs/config.yaml",
        "--seed", "42",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Quick training test passed")
            return True
        else:
            print(f"✗ Quick training test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Quick training test timed out")
        return False
    except Exception as e:
        print(f"✗ Quick training test error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for Member 3 experimental framework"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip the quick training test"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (imports, config, syntax)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MEMBER 3 EXPERIMENTAL FRAMEWORK SMOKE TEST")
    print("="*60)
    
    all_passed = True
    
    # Quick tests
    all_passed &= test_imports()
    all_passed &= test_config_loading()
    all_passed &= test_script_syntax()
    all_passed &= test_directory_structure()
    all_passed &= test_progress_tracker()
    
    if not args.quick:
        # Extended tests
        if not args.skip_training:
            all_passed &= run_quick_train_test()
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())