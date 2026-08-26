"""
Unified experiment orchestrator for Member 3 experimental validation.

This script provides a single entry point for running all experimental phases:
- Multi-seed validation
- Resource ablations
- Noise experiments
- Explainability analysis
- Cross-dataset generalization
- Statistical analysis

Usage:
    python experiments/run_member3_experiments.py --all
    python experiments/run_member3_experiments.py --phase multi_seed
    python experiments/run_member3_experiments.py --phase resource_ablations --model hybrid
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import progress tracker from current directory
from progress_tracker import ExperimentTracker


class ExperimentOrchestrator:
    """Orchestrates Member 3 experimental validation phases."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config["project"]["results_dir"])
        self.log_dir = self.results_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.tracker = ExperimentTracker("member3_experiments", str(self.results_dir))
        
        # Experiment tracking
        self.experiment_log = {
            "start_time": datetime.now().isoformat(),
            "config_path": config_path,
            "phases_completed": [],
            "phases_failed": [],
            "total_runtime_seconds": 0
        }
    
    def _load_config(self) -> Dict:
        """Load configuration file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _log_phase(self, phase_name: str, status: str, details: str = ""):
        """Log phase completion status."""
        log_entry = {
            "phase": phase_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        if status == "completed":
            self.experiment_log["phases_completed"].append(log_entry)
        else:
            self.experiment_log["phases_failed"].append(log_entry)
        
        print(f"\n[{status.upper()}] {phase_name}")
        if details:
            print(f"  {details}")
    
    def _save_experiment_log(self):
        """Save experiment log to file."""
        self.experiment_log["end_time"] = datetime.now().isoformat()
        
        start = datetime.fromisoformat(self.experiment_log["start_time"])
        end = datetime.fromisoformat(self.experiment_log["end_time"])
        self.experiment_log["total_runtime_seconds"] = (end - start).total_seconds()
        
        log_path = self.log_dir / f"member3_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(self.experiment_log, f, indent=2)
        
        print(f"\nExperiment log saved to {log_path}")
        return log_path
    
    def run_multi_seed_validation(
        self,
        models: List[str] = None,
        representations: List[str] = None,
        seeds: List[int] = None
    ) -> bool:
        """Run multi-seed validation experiments."""
        phase_name = "multi_seed_validation"
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
        
        models = models or ["hybrid"]
        representations = representations or ["cnn", "gnn"]
        seeds = seeds or self.config["project"].get("seeds_for_stats", [42, 1, 7])
        
        total_tasks = len(models) * len(representations)
        self.tracker.set_total_tasks(total_tasks)
        
        try:
            for model in models:
                for representation in representations:
                    task_name = f"{representation}-{model}"
                    self.tracker.start_task(task_name, {"phase": phase_name})
                    
                    cmd = [
                        sys.executable,
                        "experiments/multi_seed_runner.py",
                        "--model", model,
                        "--representation", representation,
                        "--config", self.config_path,
                    ]
                    
                    # Add seeds if specified
                    if seeds is not None:
                        cmd.extend(["--seeds"] + [str(s) for s in seeds])
                    
                    print(f"\nRunning: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        error = result.stderr or result.stdout or "Unknown error"
                        self.tracker.fail_task(task_name, error)
                        self._log_phase(
                            phase_name, "failed",
                            f"{representation}-{model} failed: {error}"
                        )
                        return False
                    else:
                        self.tracker.complete_task(task_name)
            
            self._log_phase(phase_name, "completed", f"Completed for {models} x {representations}")
            return True
            
        except Exception as e:
            self.tracker.fail_task(phase_name, str(e))
            self._log_phase(phase_name, "failed", str(e))
            return False
    
    def run_resource_ablations(
        self,
        models: List[str] = None,
        representations: List[str] = None
    ) -> bool:
        """Run resource ablation experiments."""
        phase_name = "resource_ablations"
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
        
        models = models or ["hybrid"]
        representations = representations or ["cnn", "gnn"]
        
        try:
            cmd = [
                sys.executable,
                "experiments/resource_ablations.py",
                "--config", self.config_path,
            ]
            
            for model in models:
                cmd.extend(["--models", model])
            
            for representation in representations:
                cmd.extend(["--representations", representation])
            
            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self._log_phase(phase_name, "failed", result.stderr)
                return False
            
            self._log_phase(phase_name, "completed", f"Completed for {models} x {representations}")
            return True
            
        except Exception as e:
            self._log_phase(phase_name, "failed", str(e))
            return False
    
    def run_noise_experiments(
        self,
        models: List[str] = None,
        representations: List[str] = None
    ) -> bool:
        """Run noise model experiments."""
        phase_name = "noise_experiments"
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
        
        models = models or ["hybrid"]
        representations = representations or ["cnn", "gnn"]
        
        try:
            cmd = [
                sys.executable,
                "experiments/noise_experiments.py",
                "--config", self.config_path,
            ]
            
            for model in models:
                cmd.extend(["--models", model])
            
            for representation in representations:
                cmd.extend(["--representations", representation])
            
            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self._log_phase(phase_name, "failed", result.stderr)
                return False
            
            self._log_phase(phase_name, "completed", f"Completed for {models} x {representations}")
            return True
            
        except Exception as e:
            self._log_phase(phase_name, "failed", str(e))
            return False
    
    def run_explainability(
        self,
        models: List[str] = None,
        representations: List[str] = None
    ) -> bool:
        """Run explainability analysis."""
        phase_name = "explainability"
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
        
        models = models or ["hybrid"]
        representations = representations or ["cnn", "gnn"]
        
        try:
            # CNN explainability
            if "cnn" in representations:
                cmd = [
                    sys.executable,
                    "experiments/explainability_cnn.py",
                    "--config", self.config_path,
                ]
                
                for model in models:
                    cmd.extend(["--models", model])
                
                print(f"\nRunning CNN explainability: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self._log_phase(phase_name, "failed", f"CNN explainability failed: {result.stderr}")
                    return False
            
            # GNN explainability
            if "gnn" in representations:
                cmd = [
                    sys.executable,
                    "experiments/explainability_gnn.py",
                    "--config", self.config_path,
                ]
                
                for model in models:
                    cmd.extend(["--models", model])
                
                print(f"\nRunning GNN explainability: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self._log_phase(phase_name, "failed", f"GNN explainability failed: {result.stderr}")
                    return False
            
            self._log_phase(phase_name, "completed", f"Completed for {models} x {representations}")
            return True
            
        except Exception as e:
            self._log_phase(phase_name, "failed", str(e))
            return False
    
    def run_cross_dataset(
        self,
        models: List[str] = None,
        representations: List[str] = None
    ) -> bool:
        """Run cross-dataset generalization experiments."""
        phase_name = "cross_dataset"
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
        
        models = models or ["hybrid"]
        representations = representations or ["cnn", "gnn"]
        
        try:
            cmd = [
                sys.executable,
                "experiments/cross_dataset.py",
                "--config", self.config_path,
            ]
            
            for model in models:
                cmd.extend(["--models", model])
            
            for representation in representations:
                cmd.extend(["--representations", representation])
            
            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self._log_phase(phase_name, "failed", result.stderr)
                return False
            
            self._log_phase(phase_name, "completed", f"Completed for {models} x {representations}")
            return True
            
        except Exception as e:
            self._log_phase(phase_name, "failed", str(e))
            return False
    
    def run_statistical_analysis(self) -> bool:
        """Run statistical analysis on collected results."""
        phase_name = "statistical_analysis"
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
        
        try:
            cmd = [
                sys.executable,
                "experiments/statistical_analysis.py",
                "--config", self.config_path,
            ]
            
            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self._log_phase(phase_name, "failed", result.stderr)
                return False
            
            self._log_phase(phase_name, "completed", "Statistical analysis completed")
            return True
            
        except Exception as e:
            self._log_phase(phase_name, "failed", str(e))
            return False
    
    def run_all_phases(self) -> bool:
        """Run all experimental phases in sequence."""
        print(f"\n{'='*60}")
        print("MEMBER 3 EXPERIMENTAL VALIDATION SUITE")
        print(f"{'='*60}")
        print(f"Config: {self.config_path}")
        print(f"Results directory: {self.results_dir}")
        print(f"Start time: {self.experiment_log['start_time']}")
        
        phases = [
            ("multi_seed_validation", self.run_multi_seed_validation),
            ("resource_ablations", self.run_resource_ablations),
            ("noise_experiments", self.run_noise_experiments),
            ("explainability", self.run_explainability),
            ("cross_dataset", self.run_cross_dataset),
            ("statistical_analysis", self.run_statistical_analysis),
        ]
        
        self.tracker.set_total_tasks(len(phases))
        all_success = True
        
        for phase_name, phase_func in phases:
            self.tracker.start_task(phase_name)
            try:
                success = phase_func()
                if not success:
                    all_success = False
                    print(f"\n⚠️  Phase {phase_name} failed, continuing with remaining phases...")
                    self.tracker.fail_task(phase_name)
                else:
                    self.tracker.complete_task(phase_name)
            except Exception as e:
                all_success = False
                self._log_phase(phase_name, "failed", str(e))
                self.tracker.fail_task(phase_name, str(e))
                print(f"\n⚠️  Phase {phase_name} encountered error: {e}")
        
        # Save final experiment log
        log_path = self._save_experiment_log()
        
        # Mark progress tracker as complete
        if all_success:
            self.tracker.mark_complete()
        else:
            self.tracker.mark_failed("Some phases failed")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Completed phases: {len(self.experiment_log['phases_completed'])}")
        print(f"Failed phases: {len(self.experiment_log['phases_failed'])}")
        print(f"Total runtime: {self.experiment_log['total_runtime_seconds']:.2f} seconds")
        print(f"Log saved to: {log_path}")
        
        # Print progress summary
        self.tracker.print_progress()
        
        if self.experiment_log["phases_failed"]:
            print("\nFailed phases:")
            for entry in self.experiment_log["phases_failed"]:
                print(f"  - {entry['phase']}: {entry['details']}")
        
        return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Run Member 3 experimental validation suite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=[
            "all",
            "multi_seed",
            "resource_ablations",
            "noise",
            "explainability",
            "cross_dataset",
            "statistical_analysis"
        ],
        default="all",
        help="Specific phase to run (default: all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experimental phases (same as --phase all)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to include in experiments"
    )
    parser.add_argument(
        "--representations",
        type=str,
        nargs="+",
        choices=["cnn", "gnn"],
        default=None,
        help="Representations to include (cnn, gnn, or both)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Specific seeds for multi-seed validation"
    )
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.phase = "all"
    
    orchestrator = ExperimentOrchestrator(args.config)
    
    if args.phase == "all":
        success = orchestrator.run_all_phases()
    elif args.phase == "multi_seed":
        success = orchestrator.run_multi_seed_validation(
            models=args.models,
            representations=args.representations,
            seeds=args.seeds
        )
    elif args.phase == "resource_ablations":
        success = orchestrator.run_resource_ablations(
            models=args.models,
            representations=args.representations
        )
    elif args.phase == "noise":
        success = orchestrator.run_noise_experiments(
            models=args.models,
            representations=args.representations
        )
    elif args.phase == "explainability":
        success = orchestrator.run_explainability(
            models=args.models,
            representations=args.representations
        )
    elif args.phase == "cross_dataset":
        success = orchestrator.run_cross_dataset(
            models=args.models,
            representations=args.representations
        )
    elif args.phase == "statistical_analysis":
        success = orchestrator.run_statistical_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()