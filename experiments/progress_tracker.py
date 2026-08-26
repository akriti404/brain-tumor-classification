"""
Progress tracking utility for long-running experiments.

Provides simple progress monitoring and checkpointing for experimental runs.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any


class ExperimentTracker:
    """Track progress of experimental runs with checkpointing."""
    
    def __init__(self, experiment_name: str, results_dir: str = "results"):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.progress_dir = self.results_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.progress_dir / f"{experiment_name}_progress.json"
        self.progress_data = self._load_progress()
        
        if not self.progress_data:
            self._initialize_progress()
    
    def _load_progress(self) -> Dict:
        """Load existing progress data."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _initialize_progress(self) -> None:
        """Initialize progress tracking data."""
        self.progress_data = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "initialized",
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task": None,
            "tasks": [],
            "checkpoints": [],
        }
        self._save_progress()
    
    def _save_progress(self) -> None:
        """Save current progress to file."""
        self.progress_data["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.progress_data, f, indent=2)
    
    def set_total_tasks(self, total: int) -> None:
        """Set the total number of tasks to complete."""
        self.progress_data["total_tasks"] = total
        self.progress_data["status"] = "running"
        self._save_progress()
    
    def start_task(self, task_name: str, task_info: Dict = None) -> None:
        """Mark a task as started."""
        self.progress_data["current_task"] = task_name
        task_data = {
            "name": task_name,
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "info": task_info or {},
        }
        self.progress_data["tasks"].append(task_data)
        self._save_progress()
        print(f"[{self.experiment_name}] Starting task: {task_name}")
    
    def complete_task(self, task_name: str, result: Dict = None) -> None:
        """Mark a task as completed."""
        # Find the task and update it
        for task in reversed(self.progress_data["tasks"]):
            if task["name"] == task_name and task["status"] == "started":
                task["status"] = "completed"
                task["end_time"] = datetime.now().isoformat()
                task["result"] = result or {}
                break
        
        self.progress_data["completed_tasks"] += 1
        self.progress_data["current_task"] = None
        self._save_progress()
        
        progress_pct = (self.progress_data["completed_tasks"] / 
                       max(self.progress_data["total_tasks"], 1)) * 100
        print(f"[{self.experiment_name}] Completed task: {task_name} ({progress_pct:.1f}%)")
    
    def fail_task(self, task_name: str, error: str = None) -> None:
        """Mark a task as failed."""
        # Find the task and update it
        for task in reversed(self.progress_data["tasks"]):
            if task["name"] == task_name and task["status"] == "started":
                task["status"] = "failed"
                task["end_time"] = datetime.now().isoformat()
                task["error"] = error or "Unknown error"
                break
        
        self.progress_data["failed_tasks"] += 1
        self.progress_data["current_task"] = None
        self._save_progress()
        print(f"[{self.experiment_name}] Failed task: {task_name}")
    
    def save_checkpoint(self, checkpoint_name: str, data: Dict = None) -> None:
        """Save a checkpoint with experiment state."""
        checkpoint_data = {
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
            "progress": {
                "completed": self.progress_data["completed_tasks"],
                "total": self.progress_data["total_tasks"],
            }
        }
        self.progress_data["checkpoints"].append(checkpoint_data)
        self._save_progress()
        print(f"[{self.experiment_name}] Checkpoint saved: {checkpoint_name}")
    
    def get_progress_summary(self) -> Dict:
        """Get a summary of current progress."""
        if not self.progress_data:
            return {}
        
        start_time = datetime.fromisoformat(self.progress_data["start_time"])
        elapsed = datetime.now() - start_time
        
        summary = {
            "experiment": self.experiment_name,
            "status": self.progress_data["status"],
            "elapsed_time": str(elapsed),
            "progress": f"{self.progress_data['completed_tasks']}/{self.progress_data['total_tasks']}",
            "percentage": (self.progress_data["completed_tasks"] / 
                          max(self.progress_data["total_tasks"], 1)) * 100,
            "failed": self.progress_data["failed_tasks"],
            "current_task": self.progress_data["current_task"],
            "checkpoints": len(self.progress_data["checkpoints"]),
        }
        
        return summary
    
    def print_progress(self) -> None:
        """Print current progress to console."""
        summary = self.get_progress_summary()
        if not summary:
            print("No progress data available")
            return
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT PROGRESS: {summary['experiment']}")
        print(f"{'='*60}")
        print(f"Status: {summary['status']}")
        print(f"Progress: {summary['progress']} ({summary['percentage']:.1f}%)")
        print(f"Elapsed: {summary['elapsed_time']}")
        print(f"Failed tasks: {summary['failed']}")
        print(f"Checkpoints: {summary['checkpoints']}")
        if summary['current_task']:
            print(f"Current task: {summary['current_task']}")
        print(f"{'='*60}\n")
    
    def mark_complete(self) -> None:
        """Mark the entire experiment as complete."""
        self.progress_data["status"] = "completed"
        self.progress_data["end_time"] = datetime.now().isoformat()
        self.progress_data["current_task"] = None
        self._save_progress()
        print(f"[{self.experiment_name}] Experiment completed successfully")
    
    def mark_failed(self, error: str = None) -> None:
        """Mark the entire experiment as failed."""
        self.progress_data["status"] = "failed"
        self.progress_data["end_time"] = datetime.now().isoformat()
        self.progress_data["error"] = error or "Unknown error"
        self.progress_data["current_task"] = None
        self._save_progress()
        print(f"[{self.experiment_name}] Experiment failed: {error}")


def estimate_remaining_time(tracker: ExperimentTracker) -> str:
    """Estimate remaining time based on current progress."""
    summary = tracker.get_progress_summary()
    
    if summary["percentage"] == 0:
        return "Unknown (no progress yet)"
    
    if summary["percentage"] >= 100:
        return "Complete"
    
    start_time = datetime.fromisoformat(tracker.progress_data["start_time"])
    elapsed = datetime.now() - start_time
    
    # Simple linear extrapolation
    remaining_fraction = (100 - summary["percentage"]) / 100
    estimated_remaining = elapsed * remaining_fraction / summary["percentage"]
    
    return str(estimated_remaining)


def cleanup_old_progress(days_old: int = 7, results_dir: str = "results") -> int:
    """Clean up progress files older than specified days."""
    progress_dir = Path(results_dir) / "progress"
    if not progress_dir.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(days=days_old)
    cleaned = 0
    
    for progress_file in progress_dir.glob("*_progress.json"):
        file_time = datetime.fromtimestamp(progress_file.stat().st_mtime)
        if file_time < cutoff_time:
            progress_file.unlink()
            cleaned += 1
    
    return cleaned


def main():
    """Simple command-line interface for progress tracking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment progress tracking utility")
    parser.add_argument("--experiment", type=str, help="Experiment name to check")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--cleanup", type=int, help="Clean up progress files older than N days")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleaned = cleanup_old_progress(args.cleanup, args.results_dir)
        print(f"Cleaned up {cleaned} old progress files")
    elif args.experiment:
        tracker = ExperimentTracker(args.experiment, args.results_dir)
        tracker.print_progress()
        
        remaining = estimate_remaining_time(tracker)
        print(f"Estimated remaining time: {remaining}")
    else:
        # List all experiments
        progress_dir = Path(args.results_dir) / "progress"
        if progress_dir.exists():
            print("Available experiments:")
            for progress_file in progress_dir.glob("*_progress.json"):
                print(f"  - {progress_file.stem.replace('_progress', '')}")
        else:
            print("No progress files found")


if __name__ == "__main__":
    main()