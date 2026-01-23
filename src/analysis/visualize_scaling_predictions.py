#!/usr/bin/env python3
"""
Visualize neural scaling law predictions from saved evaluation results.

Commands:
    all         Run all visualization commands at once
    compare     Compare MAE across multiple conditions (directories)
    trajectory  Plot prediction trajectories (compare conditions/models)
    aggregate   Show aggregate MAE across all models
    errors      Plot average error per step across methods

Usage:
    # Run all visualizations for multiple conditions
    python -m src.analysis.visualize_scaling_predictions all logistic neural

    # Compare MAE using shorthand condition names
    python -m src.analysis.visualize_scaling_predictions compare logistic neural

    # Plot trajectories comparing conditions for one model
    python -m src.analysis.visualize_scaling_predictions trajectory logistic neural --model DataDecide-c4-300M

    # Show aggregate MAE for multiple conditions
    python -m src.analysis.visualize_scaling_predictions aggregate logistic neural probe

    # Plot average error per step across methods
    python -m src.analysis.visualize_scaling_predictions errors logistic baseline

Available shortcuts: logistic, neural, baseline, probe, delta
"""

# Known condition shortcuts -> (label, path)
# add your own conditions here for easy loading
CONDITION_SHORTCUTS = {
    # "logistic": ("Logistic", "./results/dataset_test/logistic"),
}

# Standard color palette (Set2 from matplotlib)
import matplotlib.cm as cm

_SET2_CMAP = cm.get_cmap("Set2")
COLORS = [_SET2_CMAP(i) for i in range(8)]  # Set2 has 8 colors

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.eval_output import EvalPredictions, EvalResults


def filter_steps_mask(
    steps: np.ndarray, step_min: Optional[int], step_max: Optional[int]
) -> np.ndarray:
    """Return a boolean mask for steps within the given range."""
    mask = np.ones(len(steps), dtype=bool)
    if step_min is not None:
        mask &= steps >= step_min
    if step_max is not None:
        mask &= steps <= step_max
    return mask


def compute_mae_from_predictions(
    preds: EvalPredictions,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute MAE per task from predictions, optionally filtering by step range.

    Args:
        preds: EvalPredictions object
        step_min: Minimum step to include
        step_max: Maximum step to include

    Returns:
        Dict mapping task name to MAE
    """
    task_maes = {}
    for task_name, task_data in preds.task_predictions.items():
        if not task_data.errors_per_step or not task_data.steps:
            task_maes[task_name] = float("nan")
            continue

        steps = np.array(task_data.steps)
        errors = np.array(task_data.errors_per_step)

        # Handle context_end_idx - errors_per_step only covers prediction region
        ctx_end = task_data.context_end_idx
        if ctx_end is not None and ctx_end + 1 < len(steps):
            pred_steps = steps[ctx_end + 1 : ctx_end + 1 + len(errors)]
        else:
            pred_steps = steps[: len(errors)]

        # Apply step filter
        mask = filter_steps_mask(pred_steps, step_min, step_max)
        if mask.sum() == 0:
            task_maes[task_name] = float("nan")
            continue

        filtered_errors = errors[: len(mask)][mask]
        task_maes[task_name] = float(np.mean(np.abs(filtered_errors)))

    return task_maes


def load_predictions_with_mae(
    eval_dir: Path,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load predictions and compute MAE per task, optionally filtering by step range.

    Returns: {model_name: {task: mae}}
    """
    seed_dirs = discover_seed_dirs(str(eval_dir))

    if not seed_dirs:
        # No seed dirs, load directly
        results = {}
        model_names = discover_models(str(eval_dir))
        for model_name in model_names:
            preds = load_predictions_from_dir(eval_dir, model_name)
            if preds:
                results[model_name] = compute_mae_from_predictions(
                    preds, step_min, step_max
                )
        return results

    # Collect per-model, per-task MAE across seeds
    model_task_maes: Dict[str, Dict[str, List[float]]] = {}

    for seed_dir in seed_dirs:
        model_names = discover_models(str(seed_dir))
        for model_name in model_names:
            preds = load_predictions_from_dir(seed_dir, model_name)
            if preds is None:
                continue

            task_maes = compute_mae_from_predictions(preds, step_min, step_max)

            if model_name not in model_task_maes:
                model_task_maes[model_name] = {}
            for task, mae in task_maes.items():
                if task not in model_task_maes[model_name]:
                    model_task_maes[model_name][task] = []
                if not np.isnan(mae):
                    model_task_maes[model_name][task].append(mae)

    # Average across seeds
    averaged = {}
    for model_name, task_maes in model_task_maes.items():
        averaged[model_name] = {
            task: np.mean(maes) if maes else float("nan")
            for task, maes in task_maes.items()
        }

    return averaged


def discover_models(eval_dir: str) -> List[str]:
    """Discover all model names in the eval directory."""
    eval_path = Path(eval_dir)
    model_names = set()
    for f in eval_path.glob("*_results.json"):
        model_name = f.stem.replace("_results", "")
        model_names.add(model_name)
    size_order = {"90M": 0, "150M": 1, "300M": 2, "530M": 3, "750M": 4, "1B": 5}
    return sorted(model_names, key=lambda x: size_order.get(x.split("-")[-1], 99))


def discover_seed_dirs(eval_dir: str) -> List[Path]:
    """Discover seed subdirectories in the eval directory."""
    eval_path = Path(eval_dir)
    seed_dirs = []
    for d in sorted(eval_path.iterdir()):
        if d.is_dir() and d.name.startswith("seed"):
            if list(d.glob("*_results.json")):
                seed_dirs.append(d)
    return seed_dirs


def load_results_from_dir(eval_dir: Path) -> Dict[str, EvalResults]:
    """Load all EvalResults from a directory."""
    results = {}
    for model_name in discover_models(str(eval_dir)):
        results_path = eval_dir / f"{model_name}_results.json"
        if results_path.exists():
            results[model_name] = EvalResults.load(results_path)
    return results


def load_results_averaged_across_seeds(eval_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Load results from multiple seed directories and average MAE per task.

    Returns: {model_name: {task: mean_mae}}
    """
    seed_dirs = discover_seed_dirs(str(eval_dir))

    if not seed_dirs:
        # No seed dirs, load directly
        results = load_results_from_dir(eval_dir)
        averaged = {}
        for model_name, eval_results in results.items():
            averaged[model_name] = {r.task: r.mae for r in eval_results.results}
        return averaged

    # Collect per-model, per-task MAE across seeds
    model_task_maes: Dict[str, Dict[str, List[float]]] = {}

    for seed_dir in seed_dirs:
        results = load_results_from_dir(seed_dir)
        for model_name, eval_results in results.items():
            if model_name not in model_task_maes:
                model_task_maes[model_name] = {}
            for task_result in eval_results.results:
                if task_result.task not in model_task_maes[model_name]:
                    model_task_maes[model_name][task_result.task] = []
                if not np.isnan(task_result.mae):
                    model_task_maes[model_name][task_result.task].append(
                        task_result.mae
                    )

    # Average
    averaged = {}
    for model_name, task_maes in model_task_maes.items():
        averaged[model_name] = {
            task: np.mean(maes) if maes else float("nan")
            for task, maes in task_maes.items()
        }

    return averaged


def load_predictions_from_dir(
    eval_dir: Path, model_name: str
) -> Optional[EvalPredictions]:
    """Load EvalPredictions for a model from a directory."""
    predictions_path = eval_dir / f"{model_name}_predictions.json"
    if predictions_path.exists():
        return EvalPredictions.load(predictions_path)
    return None


class ScalingViz:
    """Visualization commands for scaling law predictions."""

    def compare(
        self,
        *conditions: str,
        eval_dirs: Optional[Dict[str, str]] = None,
        output_dir: str = "./results/scaling_viz",
        tasks: Optional[List[str]] = None,
        aggregate_models: bool = True,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ):
        """
        Compare MAE across multiple evaluation conditions.

        Args:
            *conditions: Shorthand condition names (e.g., "logistic", "neural", "probe")
                        Known shortcuts: logistic, neural, baseline, probe, delta
            eval_dirs: Dict mapping condition_name -> eval_dir path (alternative to shortcuts)
            output_dir: Output directory for visualizations
            tasks: Tasks to include (default: all common tasks)
            aggregate_models: If True, average MAE across models per task
            step_min: Minimum step to include in MAE computation (default: no minimum)
            step_max: Maximum step to include in MAE computation (default: no maximum)

        Examples:
            # Using shortcuts (easiest)
            python -m src.analysis.visualize_scaling_predictions compare logistic neural

            # Using explicit dict
            python -m src.analysis.visualize_scaling_predictions compare \\
                --eval_dirs '{"Logistic": "./results/scaling_eval/logistic", "Neural": "./results/scaling_eval/neural"}'

            # Mix: shortcuts with custom path override
            python -m src.analysis.visualize_scaling_predictions compare logistic \\
                --eval_dirs '{"My Neural": "./results/custom_neural"}'

            # Filter by step range
            python -m src.analysis.visualize_scaling_predictions compare logistic neural --step_max 120000
        """
        # Build eval_dirs from shortcuts and/or explicit dict
        resolved_dirs: Dict[str, str] = {}

        # First, resolve shortcuts
        for cond in conditions:
            if cond in CONDITION_SHORTCUTS:
                label, path = CONDITION_SHORTCUTS[cond]
                resolved_dirs[label] = path
            else:
                # Assume it's a path, use basename as label
                resolved_dirs[Path(cond).name] = cond

        # Then, add/override with explicit eval_dirs
        if eval_dirs:
            resolved_dirs.update(eval_dirs)

        if not resolved_dirs:
            print(
                "No conditions specified. Use shortcuts like 'logistic neural' or --eval_dirs"
            )
            print(f"Available shortcuts: {', '.join(CONDITION_SHORTCUTS.keys())}")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Print step range info
        if step_min is not None or step_max is not None:
            range_str = f"[{step_min or 0}, {step_max or 'inf'}]"
            print(f"Filtering to step range: {range_str}")

        # Load results for each condition
        condition_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for cond_name, eval_dir in resolved_dirs.items():
            eval_path = Path(eval_dir)
            print(f"Loading {cond_name} from {eval_dir}...")
            # Use step-filtered loading if step range is specified
            if step_min is not None or step_max is not None:
                condition_results[cond_name] = load_predictions_with_mae(
                    eval_path, step_min, step_max
                )
            else:
                condition_results[cond_name] = load_results_averaged_across_seeds(
                    eval_path
                )

        if not condition_results:
            print("No results loaded")
            return

        # Find common models and tasks
        all_models = set.intersection(
            *[set(r.keys()) for r in condition_results.values()]
        )
        if not all_models:
            print("No common models found across conditions")
            return

        all_tasks = set()
        for cond_results in condition_results.values():
            for model_results in cond_results.values():
                all_tasks.update(model_results.keys())

        if tasks:
            all_tasks = all_tasks & set(tasks)

        all_tasks = sorted(all_tasks)
        print(f"Common models: {sorted(all_models)}")
        print(f"Tasks: {len(all_tasks)}")

        if aggregate_models:
            # Aggregate across models: condition -> {task -> mean_mae}
            cond_task_maes: Dict[str, Dict[str, float]] = {}
            for cond_name, cond_results in condition_results.items():
                task_maes: Dict[str, List[float]] = {}
                for model_name in all_models:
                    if model_name not in cond_results:
                        continue
                    for task, mae in cond_results[model_name].items():
                        if task not in all_tasks:
                            continue
                        if task not in task_maes:
                            task_maes[task] = []
                        if not np.isnan(mae):
                            task_maes[task].append(mae)
                cond_task_maes[cond_name] = {
                    task: np.mean(maes) if maes else float("nan")
                    for task, maes in task_maes.items()
                }

            # Plot
            fig_path = output_path / "mae_comparison.png"
            self._plot_mae_bar_chart(cond_task_maes, all_tasks, str(fig_path))
        else:
            # Per-model comparison
            for model_name in sorted(all_models):
                cond_task_maes = {}
                for cond_name, cond_results in condition_results.items():
                    if model_name in cond_results:
                        cond_task_maes[cond_name] = cond_results[model_name]

                fig_path = output_path / f"mae_comparison_{model_name}.png"
                self._plot_mae_bar_chart(
                    cond_task_maes,
                    all_tasks,
                    str(fig_path),
                    title=f"MAE Comparison: {model_name}",
                )

        print(f"\nVisualizations saved to {output_dir}")

    def _plot_mae_bar_chart(
        self,
        conditions: Dict[str, Dict[str, float]],
        tasks: List[str],
        output_path: str,
        title: str = "MAE Comparison",
    ):
        """Create bar chart comparing MAE across conditions per task."""
        condition_names = list(conditions.keys())

        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(tasks))
        n_conditions = len(condition_names)
        width = 0.8 / n_conditions

        for i, cond_name in enumerate(condition_names):
            cond_data = conditions[cond_name]
            maes = [cond_data.get(t, float("nan")) for t in tasks]
            offset = (i - (n_conditions - 1) / 2) * width
            ax.bar(
                x + offset,
                maes,
                width,
                label=cond_name,
                color=COLORS[i % len(COLORS)],
                alpha=0.8,
            )

        ax.set_ylabel("Mean Absolute Error", fontsize=14)
        ax.set_xlabel("Task", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=90, ha="center", fontsize=8)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

        # Print summary
        self._print_summary(conditions, tasks, condition_names)

    def _print_summary(
        self,
        conditions: Dict[str, Dict[str, float]],
        tasks: List[str],
        condition_names: List[str],
    ):
        """Print summary table."""
        col_width = max(12, max(len(c) for c in condition_names) + 2)
        header_width = 40 + col_width * len(condition_names)

        print("\n" + "=" * header_width)
        print("SUMMARY: MAE by Task")
        print("=" * header_width)

        header = f"{'Task':<40}"
        for cond_name in condition_names:
            header += f"{cond_name:>{col_width}}"
        print(header)
        print("-" * header_width)

        for task in tasks:
            row = f"{task:<40}"
            for cond_name in condition_names:
                mae = conditions[cond_name].get(task, float("nan"))
                row += f"{mae:>{col_width}.4f}"
            print(row)

        print("-" * header_width)
        row = f"{'Overall Mean':<40}"
        for cond_name in condition_names:
            mean_mae = np.nanmean(
                [conditions[cond_name].get(t, float("nan")) for t in tasks]
            )
            row += f"{mean_mae:>{col_width}.4f}"
        print(row)
        print("=" * header_width)

    def trajectory(
        self,
        *conditions: str,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        output_dir: str = "./results/scaling_viz",
        tasks: Optional[List[str]] = None,
        eval_dirs: Optional[Dict[str, str]] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ):
        """
        Plot prediction trajectories comparing multiple conditions/models.

        Args:
            *conditions: Shorthand condition names (e.g., "logistic", "neural", "probe")
                        Known shortcuts: logistic, neural, baseline, probe, delta
            model: Single model name (e.g., DataDecide-c4-300M)
            models: List of model names to compare
            output_dir: Output directory for visualizations
            tasks: Tasks to visualize (default: all tasks)
            eval_dirs: Dict mapping condition_name -> eval_dir path (alternative to shortcuts)
            step_min: Minimum step to visualize (default: no minimum)
            step_max: Maximum step to visualize (default: no maximum)

        Examples:
            # Single condition, single model
            python -m src.analysis.visualize_scaling_predictions trajectory neural --model DataDecide-c4-300M

            # Compare conditions for one model
            python -m src.analysis.visualize_scaling_predictions trajectory logistic neural --model DataDecide-c4-300M

            # Compare multiple models for one condition
            python -m src.analysis.visualize_scaling_predictions trajectory neural --models '["DataDecide-c4-300M", "DataDecide-c4-530M"]'

            # Filter by step range
            python -m src.analysis.visualize_scaling_predictions trajectory neural --step_max 120000
        """
        # Build resolved_dirs from shortcuts
        resolved_dirs: Dict[str, str] = {}
        for cond in conditions:
            if cond in CONDITION_SHORTCUTS:
                label, path = CONDITION_SHORTCUTS[cond]
                resolved_dirs[label] = path
            else:
                resolved_dirs[Path(cond).name] = cond

        # Add/override with explicit eval_dirs
        if eval_dirs:
            resolved_dirs.update(eval_dirs)

        if not resolved_dirs:
            print(
                "No conditions specified. Use shortcuts like 'neural logistic' or --eval_dirs"
            )
            print(f"Available shortcuts: {', '.join(CONDITION_SHORTCUTS.keys())}")
            return

        # Resolve models
        model_names = models or ([model] if model else None)
        if not model_names:
            # Auto-discover from first condition
            first_dir = Path(list(resolved_dirs.values())[0])
            seed_dirs = discover_seed_dirs(str(first_dir))
            if seed_dirs:
                model_names = discover_models(str(seed_dirs[0]))
            else:
                model_names = discover_models(str(first_dir))
            if not model_names:
                print("No models found. Specify with --model or --models")
                return
            print(f"Auto-discovered models: {model_names}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load predictions for each condition
        # Structure: {cond_name: {model_name: EvalPredictions}}
        all_predictions: Dict[str, Dict[str, EvalPredictions]] = {}
        for cond_name, cond_dir in resolved_dirs.items():
            cond_path = Path(cond_dir)
            seed_dirs = discover_seed_dirs(str(cond_path))
            # Use first seed dir if available, else use cond_path directly
            load_path = seed_dirs[0] if seed_dirs else cond_path

            all_predictions[cond_name] = {}
            for mn in model_names:
                preds = load_predictions_from_dir(load_path, mn)
                if preds:
                    all_predictions[cond_name][mn] = preds

        # Find common tasks across all loaded predictions
        task_sets = []
        for cond_preds in all_predictions.values():
            for preds in cond_preds.values():
                task_sets.append(set(preds.task_predictions.keys()))

        if not task_sets:
            print("No predictions found")
            return

        common_tasks = set.intersection(*task_sets) if task_sets else set()
        if tasks:
            common_tasks = common_tasks & set(tasks)
        task_names = sorted(common_tasks)

        if not task_names:
            print("No common tasks found across conditions/models")
            return

        print(
            f"Visualizing {len(task_names)} tasks for {len(model_names)} models across {len(resolved_dirs)} conditions"
        )

        # Print step range info
        if step_min is not None or step_max is not None:
            range_str = f"[{step_min or 0}, {step_max or 'inf'}]"
            print(f"Filtering to step range: {range_str}")

        cond_names = list(resolved_dirs.keys())

        # Create one figure per model
        for model_name in model_names:
            n_tasks = len(task_names)
            n_cols = min(4, n_tasks)
            n_rows = (n_tasks + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, task_name in enumerate(task_names):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                # Find ground truth from first condition that has this task
                gt_steps = None
                gt_accs = None
                gt_ctx_end = None
                for cond_name in cond_names:
                    if model_name in all_predictions[cond_name]:
                        task_data = all_predictions[cond_name][
                            model_name
                        ].task_predictions.get(task_name)
                        if task_data and task_data.steps:
                            gt_steps = np.array(task_data.steps)
                            gt_accs = np.array(task_data.accuracies)
                            gt_ctx_end = task_data.context_end_idx
                            break

                if gt_steps is None:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(task_name, fontsize=10)
                    continue

                # Apply step filter to ground truth
                gt_mask = filter_steps_mask(gt_steps, step_min, step_max)
                gt_steps_filtered = gt_steps[gt_mask]
                gt_accs_filtered = gt_accs[gt_mask]

                # Plot ground truth (filtered points)
                ax.plot(
                    gt_steps_filtered,
                    gt_accs_filtered,
                    "ko-",
                    linewidth=1.5,
                    markersize=4,
                    label="Ground Truth",
                )

                # Plot predictions for each condition
                for ci, cond_name in enumerate(cond_names):
                    if model_name not in all_predictions[cond_name]:
                        continue
                    cond_task_data = all_predictions[cond_name][
                        model_name
                    ].task_predictions.get(task_name)
                    if not cond_task_data:
                        continue

                    preds = cond_task_data.predictions
                    cond_steps = np.array(cond_task_data.steps)
                    cond_ctx_end = cond_task_data.context_end_idx
                    color = COLORS[ci % len(COLORS)]

                    if preds:
                        medians = np.array(
                            [p.get("median", p.get("q50", 0)) for p in preds]
                        )
                        n_preds = len(medians)

                        # Determine prediction steps based on context_end_idx
                        if cond_ctx_end is not None:
                            # Predictions start after context_end_idx
                            pred_steps = cond_steps[
                                cond_ctx_end + 1 : cond_ctx_end + 1 + n_preds
                            ]
                        else:
                            # No context split (e.g., logistic) - predictions cover all steps
                            pred_steps = cond_steps[:n_preds]

                        # Only plot if we have matching steps
                        if len(pred_steps) == n_preds:
                            # Apply step filter to predictions
                            pred_mask = filter_steps_mask(
                                pred_steps, step_min, step_max
                            )
                            pred_steps_filtered = pred_steps[pred_mask]
                            medians_filtered = medians[pred_mask]

                            ax.plot(
                                pred_steps_filtered,
                                medians_filtered,
                                "-",
                                color=color,
                                linewidth=1.5,
                                alpha=0.9,
                                label=cond_name,
                            )

                            if preds[0] and "q25" in preds[0]:
                                q25s = np.array([p["q25"] for p in preds])
                                q75s = np.array([p["q75"] for p in preds])
                                ax.fill_between(
                                    pred_steps_filtered,
                                    q25s[pred_mask],
                                    q75s[pred_mask],
                                    color=color,
                                    alpha=0.1,
                                )

                        # Draw context boundary if available and within step range
                        if cond_ctx_end is not None and cond_ctx_end < len(cond_steps):
                            ctx_step = cond_steps[cond_ctx_end]
                            if (step_min is None or ctx_step >= step_min) and (
                                step_max is None or ctx_step <= step_max
                            ):
                                ax.axvline(
                                    x=ctx_step, color=color, linestyle=":", alpha=0.3
                                )

                ax.set_title(task_name, fontsize=10)
                ax.grid(True, alpha=0.3)

                if idx == 0:
                    ax.legend(fontsize=8, loc="lower right")

            # Hide empty subplots
            for idx in range(n_tasks, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].set_visible(False)

            plt.suptitle(f"Predictions: {model_name}", fontsize=14, y=1.02)
            plt.tight_layout()

            fig_path = output_path / f"{model_name}_trajectories.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {fig_path}")
            plt.close()

    def aggregate(
        self,
        *conditions: str,
        eval_dir: Optional[str] = None,
        output_dir: str = "./results/scaling_viz",
        tasks: Optional[List[str]] = None,
        eval_dirs: Optional[Dict[str, str]] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ):
        """
        Show aggregate MAE across all models for one or more conditions.

        Args:
            *conditions: Shorthand condition names (e.g., "logistic", "neural", "probe")
                        Known shortcuts: logistic, neural, baseline, probe, delta
            eval_dir: Override path for single condition (alternative to shortcuts)
            output_dir: Output directory for visualizations
            tasks: Tasks to include (default: all tasks)
            eval_dirs: Dict mapping condition_name -> eval_dir path (alternative to shortcuts)
            step_min: Minimum step to include in MAE computation (default: no minimum)
            step_max: Maximum step to include in MAE computation (default: no maximum)

        Examples:
            # Single condition using shorthand
            python -m src.analysis.visualize_scaling_predictions aggregate neural

            # Multiple conditions
            python -m src.analysis.visualize_scaling_predictions aggregate logistic neural probe

            # Using explicit path
            python -m src.analysis.visualize_scaling_predictions aggregate \\
                --eval_dir ./results/scaling_eval/neural

            # Filter by step range
            python -m src.analysis.visualize_scaling_predictions aggregate neural --step_max 120000
        """
        # Build resolved_dirs from shortcuts and/or explicit eval_dir
        resolved_dirs: Dict[str, str] = {}

        for cond in conditions:
            if cond in CONDITION_SHORTCUTS:
                label, path = CONDITION_SHORTCUTS[cond]
                resolved_dirs[label] = path
            else:
                # Assume it's a path, use basename as label
                resolved_dirs[Path(cond).name] = cond

        if eval_dir:
            resolved_dirs[Path(eval_dir).name] = eval_dir

        # Add/override with explicit eval_dirs
        if eval_dirs:
            resolved_dirs.update(eval_dirs)

        if not resolved_dirs:
            print(
                "No conditions specified. Use shortcuts like 'neural' or --eval_dir/--eval_dirs"
            )
            print(f"Available shortcuts: {', '.join(CONDITION_SHORTCUTS.keys())}")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Print step range info
        if step_min is not None or step_max is not None:
            range_str = f"[{step_min or 0}, {step_max or 'inf'}]"
            print(f"Filtering to step range: {range_str}")

        # Load results for each condition
        condition_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for cond_name, cond_dir in resolved_dirs.items():
            print(f"Loading {cond_name} from {cond_dir}...")
            # Use step-filtered loading if step range is specified
            if step_min is not None or step_max is not None:
                condition_results[cond_name] = load_predictions_with_mae(
                    Path(cond_dir), step_min, step_max
                )
            else:
                condition_results[cond_name] = load_results_averaged_across_seeds(
                    Path(cond_dir)
                )

        if not condition_results:
            print("No results loaded")
            return

        # Find all models across conditions
        all_models = set()
        for cond_results in condition_results.values():
            all_models.update(cond_results.keys())
        all_models = sorted(all_models)

        print(
            f"Found {len(all_models)} models across {len(condition_results)} conditions"
        )

        # Compute per-model mean MAE for each condition
        condition_model_maes: Dict[str, Dict[str, float]] = {}
        for cond_name, cond_results in condition_results.items():
            model_maes = {}
            for model_name in all_models:
                if model_name not in cond_results:
                    model_maes[model_name] = float("nan")
                    continue
                task_maes = cond_results[model_name]
                if tasks:
                    task_maes = {t: m for t, m in task_maes.items() if t in tasks}
                maes = [m for m in task_maes.values() if not np.isnan(m)]
                model_maes[model_name] = np.mean(maes) if maes else float("nan")
            condition_model_maes[cond_name] = model_maes

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(all_models))
        n_conditions = len(condition_model_maes)
        width = 0.8 / n_conditions

        for i, (cond_name, model_maes) in enumerate(condition_model_maes.items()):
            maes = [model_maes[m] for m in all_models]
            offset = (i - (n_conditions - 1) / 2) * width
            ax.bar(
                x + offset,
                maes,
                width,
                label=cond_name,
                color=COLORS[i % len(COLORS)],
                alpha=0.8,
            )

        ax.set_ylabel("Mean MAE (across tasks)", fontsize=14)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_title("Mean MAE by Model", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("-")[-1] for m in all_models])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        fig_path = output_path / "aggregate_mae.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
        plt.close()

        # Print summary
        cond_names = list(condition_model_maes.keys())
        col_width = max(12, max(len(c) for c in cond_names) + 2)
        header_width = 30 + col_width * len(cond_names)

        print("\n" + "=" * header_width)
        print("SUMMARY: Mean MAE by Model")
        print("=" * header_width)

        header = f"{'Model':<30}"
        for cond_name in cond_names:
            header += f"{cond_name:>{col_width}}"
        print(header)
        print("-" * header_width)

        for model_name in all_models:
            row = f"{model_name:<30}"
            for cond_name in cond_names:
                mae = condition_model_maes[cond_name][model_name]
                row += f"{mae:>{col_width}.4f}"
            print(row)

        print("-" * header_width)
        row = f"{'Overall Mean':<30}"
        for cond_name in cond_names:
            mean_mae = np.nanmean(list(condition_model_maes[cond_name].values()))
            row += f"{mean_mae:>{col_width}.4f}"
        print(row)
        print("=" * header_width)

    def errors(
        self,
        *conditions: str,
        output_dir: str = "./results/scaling_viz",
        models: Optional[List[str]] = None,
        eval_dirs: Optional[Dict[str, str]] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ):
        """
        Plot average error per step across methods.

        For each model size, averages the errors_per_step vectors across all tasks
        and compares them across methods.

        Args:
            *conditions: Shorthand condition names (e.g., "logistic", "baseline")
                        Known shortcuts: logistic, neural, baseline, probe, delta
            output_dir: Output directory for visualizations
            models: List of model names to include (default: all models)
            eval_dirs: Dict mapping condition_name -> eval_dir path (alternative to shortcuts)
            step_min: Minimum step to include in visualization (default: no minimum)
            step_max: Maximum step to include in visualization (default: no maximum)

        Examples:
            # Compare two methods
            python -m src.analysis.visualize_scaling_predictions errors logistic baseline

            # Compare multiple methods
            python -m src.analysis.visualize_scaling_predictions errors logistic baseline neural

            # Filter by step range
            python -m src.analysis.visualize_scaling_predictions errors logistic neural --step_max 120000
        """
        # Build resolved_dirs from shortcuts
        resolved_dirs: Dict[str, str] = {}
        for cond in conditions:
            if cond in CONDITION_SHORTCUTS:
                label, path = CONDITION_SHORTCUTS[cond]
                resolved_dirs[label] = path
            else:
                resolved_dirs[Path(cond).name] = cond

        # Add/override with explicit eval_dirs
        if eval_dirs:
            resolved_dirs.update(eval_dirs)

        if not resolved_dirs:
            print(
                "No conditions specified. Use shortcuts like 'logistic baseline' or --eval_dirs"
            )
            print(f"Available shortcuts: {', '.join(CONDITION_SHORTCUTS.keys())}")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Discover models from first condition if not specified
        if not models:
            first_dir = Path(list(resolved_dirs.values())[0])
            seed_dirs = discover_seed_dirs(str(first_dir))
            if seed_dirs:
                models = discover_models(str(seed_dirs[0]))
            else:
                models = discover_models(str(first_dir))

        if not models:
            print("No models found")
            return

        print(f"Models: {models}")
        print(f"Conditions: {list(resolved_dirs.keys())}")

        # Print step range info
        if step_min is not None or step_max is not None:
            range_str = f"[{step_min or 0}, {step_max or 'inf'}]"
            print(f"Filtering to step range: {range_str}")

        # Load errors_per_step and steps for each condition and model
        # Structure: {cond_name: {model_name: List[(errors_per_step, pred_steps)]}}
        cond_model_errors: Dict[
            str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
        ] = {}

        for cond_name, cond_dir in resolved_dirs.items():
            cond_path = Path(cond_dir)
            seed_dirs = discover_seed_dirs(str(cond_path))
            load_path = seed_dirs[0] if seed_dirs else cond_path

            cond_model_errors[cond_name] = {}
            for model_name in models:
                preds = load_predictions_from_dir(load_path, model_name)
                if preds is None:
                    continue

                # Collect errors_per_step and steps from all tasks
                task_errors = []
                for task_name, task_data in preds.task_predictions.items():
                    if task_data.errors_per_step and task_data.steps:
                        errors = np.array(task_data.errors_per_step)
                        steps = np.array(task_data.steps)
                        ctx_end = task_data.context_end_idx

                        # Get prediction steps
                        if ctx_end is not None and ctx_end + 1 < len(steps):
                            pred_steps = steps[ctx_end + 1 : ctx_end + 1 + len(errors)]
                        else:
                            pred_steps = steps[: len(errors)]

                        if len(pred_steps) == len(errors):
                            task_errors.append((errors, pred_steps))

                if task_errors:
                    cond_model_errors[cond_name][model_name] = task_errors

        # Compute average error per step for each condition and model
        # Structure: {cond_name: {model_name: (avg_errors_per_step, pred_steps)}}
        cond_model_avg_errors: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

        for cond_name, model_errors in cond_model_errors.items():
            cond_model_avg_errors[cond_name] = {}
            for model_name, task_error_step_arrays in model_errors.items():
                # Find minimum length across all tasks (in case of different step counts)
                min_len = min(len(arr[0]) for arr in task_error_step_arrays)
                # Truncate and stack errors
                stacked = np.stack([arr[0][:min_len] for arr in task_error_step_arrays])
                # Average across tasks
                avg_errors = np.mean(stacked, axis=0)
                # Use steps from first task (they should be consistent across tasks for same model)
                pred_steps = task_error_step_arrays[0][1][:min_len]

                # Apply step filter
                mask = filter_steps_mask(pred_steps, step_min, step_max)
                cond_model_avg_errors[cond_name][model_name] = (
                    avg_errors[mask],
                    pred_steps[mask],
                )

        # Plot: one subplot per model, lines for each condition
        n_models = len(models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        cond_names = list(resolved_dirs.keys())

        for idx, model_name in enumerate(models):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            has_data = False
            for ci, cond_name in enumerate(cond_names):
                if model_name not in cond_model_avg_errors.get(cond_name, {}):
                    continue

                avg_errors, pred_steps = cond_model_avg_errors[cond_name][model_name]
                if len(avg_errors) == 0:
                    continue
                color = COLORS[ci % len(COLORS)]

                ax.plot(
                    pred_steps,
                    avg_errors,
                    "-",
                    color=color,
                    linewidth=2,
                    label=cond_name,
                    alpha=0.8,
                )
                has_data = True

            if has_data:
                ax.set_xlabel("Step", fontsize=10)
                ax.set_ylabel("Avg Error (|pred - actual|)", fontsize=10)
                ax.set_title(model_name.split("-")[-1], fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(model_name.split("-")[-1], fontsize=12)

        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle("Average Error per Step by Model Size", fontsize=14, y=1.02)
        plt.tight_layout()

        fig_path = output_path / "errors_per_step.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
        plt.close()

        # Print summary
        self._print_errors_summary(cond_model_avg_errors, models, cond_names)

    def _print_errors_summary(
        self,
        cond_model_avg_errors: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        models: List[str],
        cond_names: List[str],
    ):
        """Print summary of average errors per step."""
        col_width = max(12, max(len(c) for c in cond_names) + 2)
        header_width = 30 + col_width * len(cond_names)

        print("\n" + "=" * header_width)
        print("SUMMARY: Mean Error per Step (averaged across all steps)")
        print("=" * header_width)

        header = f"{'Model':<30}"
        for cond_name in cond_names:
            header += f"{cond_name:>{col_width}}"
        print(header)
        print("-" * header_width)

        for model_name in models:
            row = f"{model_name:<30}"
            for cond_name in cond_names:
                if model_name in cond_model_avg_errors.get(cond_name, {}):
                    avg_errors, _ = cond_model_avg_errors[cond_name][model_name]
                    if len(avg_errors) > 0:
                        mean_err = np.mean(avg_errors)
                        row += f"{mean_err:>{col_width}.4f}"
                    else:
                        row += f"{'N/A':>{col_width}}"
                else:
                    row += f"{'N/A':>{col_width}}"
            print(row)

        print("-" * header_width)
        row = f"{'Overall Mean':<30}"
        for cond_name in cond_names:
            all_means = []
            for model_name in models:
                if model_name in cond_model_avg_errors.get(cond_name, {}):
                    avg_errors, _ = cond_model_avg_errors[cond_name][model_name]
                    if len(avg_errors) > 0:
                        all_means.append(np.mean(avg_errors))
            if all_means:
                row += f"{np.mean(all_means):>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)
        print("=" * header_width)

    def all(
        self,
        *conditions: str,
        output_dir: str = "./results/scaling_viz",
        tasks: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        eval_dirs: Optional[Dict[str, str]] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ):
        """
        Run all visualization commands: compare, trajectory, aggregate, and errors.

        Args:
            *conditions: Shorthand condition names (e.g., "logistic", "neural", "probe")
                        Known shortcuts: logistic, neural, baseline, probe, delta
            output_dir: Output directory for visualizations
            tasks: Tasks to include (default: all tasks)
            models: Models to include (default: all models)
            eval_dirs: Dict mapping condition_name -> eval_dir path (alternative to shortcuts)
            step_min: Minimum step to include in MAE/visualization (default: no minimum)
            step_max: Maximum step to include in MAE/visualization (default: no maximum)

        Examples:
            # Run all visualizations for two conditions
            python -m src.analysis.visualize_scaling_predictions all logistic neural

            # Run all visualizations for multiple conditions
            python -m src.analysis.visualize_scaling_predictions all logistic neural baseline probe

            # Using explicit dict
            python -m src.analysis.visualize_scaling_predictions all \\
                --eval_dirs '{"Logistic": "./results/eval/logistic", "Neural": "./results/eval/neural"}'

            # Filter by step range (e.g., only first 120000 steps)
            python -m src.analysis.visualize_scaling_predictions all logistic neural --step_max 120000
        """
        if not conditions and not eval_dirs:
            print(
                "No conditions specified. Use shortcuts like 'logistic neural' or --eval_dirs"
            )
            print(f"Available shortcuts: {', '.join(CONDITION_SHORTCUTS.keys())}")
            return

        print("=" * 60)
        print("Running all visualizations")
        if step_min is not None or step_max is not None:
            range_str = f"[{step_min or 0}, {step_max or 'inf'}]"
            print(f"Step range filter: {range_str}")
        print("=" * 60)

        print("\n" + "-" * 60)
        print("1. COMPARE (MAE by task)")
        print("-" * 60)
        self.compare(
            *conditions,
            output_dir=output_dir,
            tasks=tasks,
            eval_dirs=eval_dirs,
            step_min=step_min,
            step_max=step_max,
        )

        print("\n" + "-" * 60)
        print("2. TRAJECTORY (prediction curves)")
        print("-" * 60)
        self.trajectory(
            *conditions,
            output_dir=output_dir,
            tasks=tasks,
            models=models,
            eval_dirs=eval_dirs,
            step_min=step_min,
            step_max=step_max,
        )

        print("\n" + "-" * 60)
        print("3. AGGREGATE (MAE by model)")
        print("-" * 60)
        self.aggregate(
            *conditions,
            output_dir=output_dir,
            tasks=tasks,
            eval_dirs=eval_dirs,
            step_min=step_min,
            step_max=step_max,
        )

        print("\n" + "-" * 60)
        print("4. ERRORS (error per step)")
        print("-" * 60)
        self.errors(
            *conditions,
            output_dir=output_dir,
            models=models,
            eval_dirs=eval_dirs,
            step_min=step_min,
            step_max=step_max,
        )

        print("\n" + "=" * 60)
        print(f"All visualizations saved to {output_dir}")
        print("=" * 60)


def main():
    fire.Fire(ScalingViz)


if __name__ == "__main__":
    main()
