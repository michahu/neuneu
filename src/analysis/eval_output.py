"""
Shared evaluation output classes for consistent JSON output format.

All evaluation scripts should use these classes to ensure consistent output.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class TaskResult:
    """Result for a single task."""
    task: str
    mae: float
    rmse: float
    num_samples: Optional[int] = None
    correlation: Optional[float] = None


@dataclass
class EvalResults:
    """Standard evaluation results container.

    All evaluation scripts should use this class to save results,
    ensuring consistent JSON output format.

    Example usage:
        results = EvalResults(
            model_name="DataDecide-c4-300M",
            model_type="neural",
            tasks=["hellaswag", "arc_easy"],
            results=[
                TaskResult(task="hellaswag", mae=0.0166, rmse=0.0234),
                TaskResult(task="arc_easy", mae=0.0189, rmse=0.0267),
            ],
            checkpoint="./results/v3/metamodel/seed0/best_model.pt",
            context_ratio=0.4,
        )
        results.save(Path("./results/scaling_eval/neural/seed0"))
    """
    model_name: str
    model_type: str  # baseline, neural, probe-cnn, probe-histogram, logistic, delta_probe
    tasks: List[str]
    results: List[TaskResult]
    checkpoint: Optional[str] = None
    context_ratio: Optional[float] = None
    seed: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

    def save(self, output_dir: Path) -> Path:
        """Save results to standard JSON format.

        Args:
            output_dir: Directory to save results to

        Returns:
            Path to the saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert TaskResult objects to dicts
        data = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "tasks": self.tasks,
            "results": [asdict(r) for r in self.results],
        }

        # Add optional fields if present
        if self.checkpoint is not None:
            data["checkpoint"] = self.checkpoint
        if self.context_ratio is not None:
            data["context_ratio"] = self.context_ratio
        if self.seed is not None:
            data["seed"] = self.seed
        if self.config is not None:
            data["config"] = self.config

        results_path = output_dir / f"{self.model_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        return results_path

    @classmethod
    def load(cls, path: Path) -> "EvalResults":
        """Load and validate results from JSON.

        Args:
            path: Path to the results JSON file

        Returns:
            EvalResults object
        """
        with open(path) as f:
            data = json.load(f)

        # Convert result dicts to TaskResult objects
        results = []
        for r in data.get("results", []):
            results.append(TaskResult(
                task=r["task"],
                mae=r["mae"],
                rmse=r["rmse"],
                num_samples=r.get("num_samples"),
                correlation=r.get("correlation"),
            ))

        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            tasks=data["tasks"],
            results=results,
            checkpoint=data.get("checkpoint"),
            context_ratio=data.get("context_ratio"),
            seed=data.get("seed"),
            config=data.get("config"),
        )

    def get_mean_mae(self) -> float:
        """Get mean MAE across all tasks."""
        import numpy as np
        maes = [r.mae for r in self.results if r.mae is not None and not np.isnan(r.mae)]
        return float(np.mean(maes)) if maes else float("nan")

    def get_mean_rmse(self) -> float:
        """Get mean RMSE across all tasks."""
        import numpy as np
        rmses = [r.rmse for r in self.results if r.rmse is not None and not np.isnan(r.rmse)]
        return float(np.mean(rmses)) if rmses else float("nan")


@dataclass
class TaskPredictions:
    """Predictions for a single task."""
    steps: List[int]
    accuracies: List[float]
    losses: List[float]
    predictions: List[Dict[str, float]]  # {"median": x, "q10": y, ...}
    context_end_idx: Optional[int] = None
    errors_per_step: Optional[List[float]] = None


@dataclass
class EvalPredictions:
    """Full predictions container.

    Stores detailed predictions for all tasks, including ground truth
    and per-step errors for visualization.
    """
    model_name: str
    model_type: str
    task_predictions: Dict[str, TaskPredictions]

    def save(self, output_dir: Path) -> Path:
        """Save predictions to standard JSON format.

        Args:
            output_dir: Directory to save predictions to

        Returns:
            Path to the saved predictions file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "tasks": {}
        }

        for task, preds in self.task_predictions.items():
            task_data = {
                "steps": preds.steps,
                "accuracies": preds.accuracies,
                "losses": preds.losses,
                "predictions": preds.predictions,
            }
            if preds.context_end_idx is not None:
                task_data["context_end_idx"] = preds.context_end_idx
            if preds.errors_per_step is not None:
                task_data["errors_per_step"] = preds.errors_per_step
            data["tasks"][task] = task_data

        predictions_path = output_dir / f"{self.model_name}_predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        return predictions_path

    @classmethod
    def load(cls, path: Path) -> "EvalPredictions":
        """Load predictions from JSON.

        Args:
            path: Path to the predictions JSON file

        Returns:
            EvalPredictions object
        """
        with open(path) as f:
            data = json.load(f)

        task_predictions = {}
        for task, preds in data.get("tasks", {}).items():
            task_predictions[task] = TaskPredictions(
                steps=preds["steps"],
                accuracies=preds["accuracies"],
                losses=preds["losses"],
                predictions=preds["predictions"],
                context_end_idx=preds.get("context_end_idx"),
                errors_per_step=preds.get("errors_per_step"),
            )

        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            task_predictions=task_predictions,
        )
