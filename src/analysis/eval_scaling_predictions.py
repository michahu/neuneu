#!/usr/bin/env python3
"""
Evaluate neural scaling law predictions and save results.

This script runs the evaluation pipeline without generating visualizations.
Results are saved to JSON/NPY files that can be visualized later.

Output files:
    {output_dir}/{model_name}_predictions.npz - All predictions and ground truth
    {output_dir}/{model_name}_results.json - Summary metrics

Usage with fire:
    # Evaluate baseline model only
    python -m src.analysis.eval_scaling_predictions baseline \
        --model_name DataDecide-c4-300M \
        --checkpoint ./results/v3/baseline/seed0/best_model.pt

    # Evaluate neural (meta-model) only
    python -m src.analysis.eval_scaling_predictions neural \
        --model_name DataDecide-c4-300M \
        --checkpoint ./results/v3/metamodel/seed0/best_model.pt

    # Evaluate probe only
    python -m src.analysis.eval_scaling_predictions probe \
        --model_name DataDecide-c4-300M \
        --checkpoint ./results/v3/probe/cnn/seed0/best_model.pt
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch

from src.analysis.eval_output import (
    EvalPredictions,
    EvalResults,
    TaskPredictions,
    TaskResult,
)
from src.analysis.utils import (
    # Error computation
    compute_errors,
    compute_errors_per_timestep,
    # Utilities
    compute_gaps,
    # Logistic fitting
    fit_logistic_scaling_law,
    get_available_tasks,
    load_avg_loss_probe_model,
    load_delta_probe_model,
    load_histograms_by_step,
    load_loss_data,
    load_model,
    load_probe_model,
    load_raw_losses,
    # Data loading
    load_step_aligned_data,
    load_training_data_for_task,
    predict_logistic,
    print_summary,
)
from src.meta.datasets.base import load_task_accuracies
from src.meta.probes import KLDeltaProbe
from src.meta.utils import _extract_step_number


@torch.no_grad()
def make_neural_prediction(
    model,
    encoder_input: np.ndarray,
    context_values: np.ndarray,
    context_gaps: np.ndarray,
    device: torch.device,
    return_quantiles: bool = True,
) -> Dict[str, float]:
    """Make a prediction using a neural scaling law model."""
    encoder_input_t = torch.from_numpy(encoder_input).unsqueeze(0).to(device)

    context_values_t = (
        torch.from_numpy(context_values.astype(np.float32)).unsqueeze(0).to(device)
    )
    context_gaps_t = (
        torch.from_numpy(context_gaps.astype(np.int64)).unsqueeze(0).to(device)
    )
    context_mask_t = torch.ones(1, len(context_values), device=device)

    output = model.predict(
        encoder_input=encoder_input_t,
        context_losses=context_values_t,
        context_gaps=context_gaps_t,
        context_mask=context_mask_t,
        return_distribution=return_quantiles,
    )

    if return_quantiles and output.dim() > 1:
        quantiles = output[0].cpu().numpy()
        return {
            "q10": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q90": float(quantiles[4]),
        }
    else:
        return {"median": float(output.item())}


def prepare_encoder_input_for_soft_prompt_type(
    soft_prompt_type: str,
    data_dir: str,
    model_name: str,
    steps: np.ndarray,
    context_indices: np.ndarray,
    all_losses: np.ndarray,
    loss_file_pattern: str = "word_losses*.npy",
    inverse_perplexity: bool = False,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    target_idx: Optional[int] = None,
    histograms: Optional[np.ndarray] = None,
    hist_step_to_idx: Optional[Dict[int, int]] = None,
    max_encoder_tokens: Optional[int] = None,
) -> np.ndarray:
    """Prepare encoder input based on soft_prompt_type.

    Args:
        soft_prompt_type: One of "cnn", "avg_loss", "delta"
        data_dir: Base data directory
        model_name: Name of the model
        steps: Array of step numbers
        context_indices: Indices into steps array for context
        all_losses: All average losses for the model
        loss_file_pattern: Glob pattern for loss files
        inverse_perplexity: If True, transform losses x -> e^(-x) (values become 0-1)
        bin_min: Minimum value for clipping (0.0 for inverse_perplexity)
        bin_max: Maximum value for clipping (1.0 for inverse_perplexity, 15.0 otherwise)
        target_idx: Target step index (required for delta soft prompt type)
        histograms: Pre-loaded histograms array (for delta type, to avoid reloading)
        hist_step_to_idx: Mapping from step number to histogram index (for delta type)
        max_encoder_tokens: Maximum number of tokens to load for CNN encoder (must match training config)

    Returns:
        Encoder input array appropriate for the soft_prompt_type
    """
    if soft_prompt_type == "avg_loss":
        # For avg_loss: encoder_input is the sequence of average losses at context steps
        return all_losses[context_indices].astype(np.float32)

    elif soft_prompt_type == "delta":
        # Delta probe: encoder_input is delta histogram (target - context_end)
        if target_idx is None:
            raise ValueError("target_idx is required for delta soft prompt type")
        if histograms is None or hist_step_to_idx is None:
            raise ValueError(
                "histograms and hist_step_to_idx are required for delta soft prompt type"
            )

        # Get step numbers
        last_context_step = int(steps[int(context_indices[-1])])
        target_step = int(steps[target_idx])

        # Get histograms and compute delta
        context_hist = histograms[hist_step_to_idx[last_context_step]]
        target_hist = histograms[hist_step_to_idx[target_step]]
        delta_histogram = target_hist - context_hist
        return delta_histogram.astype(np.float32)

    else:
        # Default: cnn - load raw losses from the last context step
        last_context_step = int(steps[int(context_indices[-1])])
        # Use max_encoder_tokens to match training config (defaults to 1M if not specified)
        max_tokens = max_encoder_tokens if max_encoder_tokens is not None else 128000
        encoder_input = load_raw_losses(
            data_dir,
            model_name,
            last_context_step,
            max_tokens=max_tokens,
            loss_file_pattern=loss_file_pattern,
            inverse_perplexity=inverse_perplexity,
            bin_min=bin_min,
            bin_max=bin_max,
        )
        if encoder_input is None:
            encoder_input = np.zeros(max_tokens, dtype=np.float32)
        return encoder_input


@torch.no_grad()
def make_sequential_predictions(
    model,
    data_dir: str,
    model_name: str,
    steps: np.ndarray,
    all_accuracies: np.ndarray,
    initial_context_end: int,
    num_predictions: int,
    device: torch.device,
    use_real_data: bool = False,
    return_quantiles: bool = True,
    loss_file_pattern: str = "word_losses*.npy",
    soft_prompt_type: str = "cnn",
    num_bins: int = 32,
    all_losses: Optional[np.ndarray] = None,
    inverse_perplexity: bool = False,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    max_encoder_tokens: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Make sequential predictions, optionally using real data at each step.

    Args:
        model: Neural meta-model
        data_dir: Base data directory
        model_name: Name of the model
        steps: Array of step numbers
        all_accuracies: All accuracy values
        initial_context_end: Index of last context observation
        num_predictions: Number of future predictions to make
        device: Device to run on
        use_real_data: Use real accuracies at each step (vs autoregressive)
        return_quantiles: Return quantile predictions
        loss_file_pattern: Glob pattern for loss files
        soft_prompt_type: Type of soft prompt generator (cnn, avg_loss, delta)
        num_bins: Number of histogram bins
        all_losses: Array of average losses (required for avg_loss/hist variants)
        inverse_perplexity: If True, transform losses x -> e^(-x) (values become 0-1)
        bin_min: Minimum value for clipping (0.0 for inverse_perplexity)
        bin_max: Maximum value for clipping (1.0 for inverse_perplexity, 15.0 otherwise)
        max_encoder_tokens: Maximum number of tokens for CNN encoder (must match training config)
    """
    predictions = []

    # Load average losses if needed for soft_prompt_type
    if all_losses is None and soft_prompt_type in ("avg_loss",):
        _, all_losses = load_loss_data(data_dir, model_name)

    # Load histograms once if needed for delta soft_prompt_type
    histograms = None
    hist_step_to_idx = None
    if soft_prompt_type == "delta":
        model_dir = Path(data_dir) / model_name
        _, histograms, hist_step_to_idx = load_histograms_by_step(
            model_dir, inverse_perplexity=inverse_perplexity
        )
        if histograms is None:
            raise ValueError(
                f"No histograms found for delta soft prompt type in {model_dir}"
            )

    context_values = all_accuracies[: initial_context_end + 1].copy()
    context_indices = np.arange(initial_context_end + 1)

    for i in range(num_predictions):
        target_idx = initial_context_end + 1 + i

        if target_idx >= len(steps):
            break

        context_gaps = compute_gaps(context_indices, target_idx)

        # Prepare encoder input based on soft_prompt_type
        encoder_input = prepare_encoder_input_for_soft_prompt_type(
            soft_prompt_type=soft_prompt_type,
            data_dir=data_dir,
            model_name=model_name,
            steps=steps,
            context_indices=context_indices,
            all_losses=all_losses if all_losses is not None else np.zeros(len(steps)),
            loss_file_pattern=loss_file_pattern,
            inverse_perplexity=inverse_perplexity,
            bin_min=bin_min,
            bin_max=bin_max,
            target_idx=target_idx,
            histograms=histograms,
            hist_step_to_idx=hist_step_to_idx,
            max_encoder_tokens=max_encoder_tokens,
        )

        pred = make_neural_prediction(
            model,
            encoder_input,
            context_values,
            context_gaps,
            device,
            return_quantiles=return_quantiles,
        )
        predictions.append(pred)

        if use_real_data:
            new_value = all_accuracies[target_idx]
        else:
            new_value = pred["median"]

        context_values = np.append(context_values, new_value)
        context_indices = np.append(context_indices, target_idx)

    return predictions


@torch.no_grad()
def make_anchored_predictions(
    model,
    data_dir: str,
    model_name: str,
    steps: np.ndarray,
    all_accuracies: np.ndarray,
    anchor_idx: int,
    num_predictions: int,
    device: torch.device,
    return_quantiles: bool = True,
    loss_file_pattern: str = "word_losses*.npy",
    soft_prompt_type: str = "cnn",
    num_bins: int = 32,
    all_losses: Optional[np.ndarray] = None,
    inverse_perplexity: bool = False,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    max_encoder_tokens: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Make anchored predictions with increasing gap from a fixed anchor point.

    Unlike sequential predictions which slide the context window forward,
    anchored predictions keep the context fixed at [0:anchor_idx+1] and
    predict increasingly distant future points.

    This is analogous to delta_probe's "anchor" mode where:
    - Y_t = Y_anchor + model(X_t - X_anchor)

    For neural/baseline models:
    - Context is fixed at steps [0:anchor_idx+1]
    - Gap increases as we predict further into the future

    Args:
        model: Neural meta-model
        data_dir: Base data directory
        model_name: Name of the model
        steps: Array of step numbers
        all_accuracies: All accuracy values
        anchor_idx: Index of the anchor point (context ends here, predictions start after)
        num_predictions: Number of future predictions to make
        device: Device to run on
        return_quantiles: Return quantile predictions
        loss_file_pattern: Glob pattern for loss files
        soft_prompt_type: Type of soft prompt generator (cnn, avg_loss, delta)
        num_bins: Number of histogram bins
        all_losses: Array of average losses (required for avg_loss variants)
        inverse_perplexity: If True, transform losses x -> e^(-x)
        bin_min: Minimum value for clipping
        bin_max: Maximum value for clipping
        max_encoder_tokens: Maximum number of tokens for CNN encoder (must match training config)

    Returns:
        List of prediction dicts, one per future timestep
    """
    predictions = []

    # Load average losses if needed for soft_prompt_type
    if all_losses is None and soft_prompt_type in ("avg_loss",):
        _, all_losses = load_loss_data(data_dir, model_name)

    # Load histograms once if needed for delta soft_prompt_type
    histograms = None
    hist_step_to_idx = None
    if soft_prompt_type == "delta":
        model_dir = Path(data_dir) / model_name
        _, histograms, hist_step_to_idx = load_histograms_by_step(
            model_dir, inverse_perplexity=inverse_perplexity
        )
        if histograms is None:
            raise ValueError(
                f"No histograms found for delta soft prompt type in {model_dir}"
            )

    context_values = all_accuracies[: anchor_idx + 1].copy()
    context_indices = np.arange(anchor_idx + 1)

    for i in range(num_predictions):
        target_idx = anchor_idx + 1 + i

        if target_idx >= len(steps):
            break

        # Compute gaps from the FIXED context to the target
        # Unlike sequential, context_indices stays fixed
        context_gaps = compute_gaps(context_indices, target_idx)

        # Prepare encoder input based on soft_prompt_type
        encoder_input = prepare_encoder_input_for_soft_prompt_type(
            soft_prompt_type=soft_prompt_type,
            data_dir=data_dir,
            model_name=model_name,
            steps=steps,
            context_indices=context_indices,
            all_losses=all_losses if all_losses is not None else np.zeros(len(steps)),
            loss_file_pattern=loss_file_pattern,
            inverse_perplexity=inverse_perplexity,
            bin_min=bin_min,
            bin_max=bin_max,
            target_idx=target_idx,
            histograms=histograms,
            hist_step_to_idx=hist_step_to_idx,
            max_encoder_tokens=max_encoder_tokens,
        )

        pred = make_neural_prediction(
            model,
            encoder_input,
            context_values,
            context_gaps,
            device,
            return_quantiles=return_quantiles,
        )
        predictions.append(pred)

    return predictions


def make_logistic_prediction(
    train_data_dir: str,
    eval_losses: np.ndarray,
    task_name: str,
) -> Tuple[List[Dict[str, float]], Optional[np.ndarray]]:
    """Make predictions using logistic fitted on training corpus (zero-shot).

    Fits a 4-parameter logistic function on all (loss, accuracy) pairs from the
    training corpus and applies it to the eval losses.

    Args:
        train_data_dir: Path to training data directory
        eval_losses: Array of loss values to predict accuracies for
        task_name: Name of the task to predict

    Returns:
        Tuple of (predictions list with "median" key, fitted parameters or None)
    """
    # Load training data for this task
    train_losses, train_accs = load_training_data_for_task(train_data_dir, task_name)

    if len(train_losses) == 0:
        print(f"    Warning: No training data found for task {task_name}")
        return [], None

    # Fit logistic on training data
    params, _ = fit_logistic_scaling_law(train_losses, train_accs)

    if params is None:
        return [], None

    # Predict on eval losses
    preds = predict_logistic(eval_losses, params)
    return [{"median": float(p)} for p in preds], params


class ScalingEvaluator:
    """CLI for evaluating scaling law predictions."""

    def __init__(
        self,
        data_dir: str = "./results/datadecide_dataset_test",
        output_dir: str = "./results/scaling_eval",
        context_ratio: float = 0.5,
        device: str = None,
        use_real_data: bool = False,
        loss_file_pattern: str = None,
        prediction_mode: str = "sequential",
    ):
        """Initialize evaluator with common settings.

        Args:
            data_dir: Directory containing evaluation data
            output_dir: Directory to save results
            context_ratio: Ratio of data to use as context (0-1)
            device: Device to use (cuda/cpu, auto-detected if None)
            use_real_data: Use real accuracies at each step (vs autoregressive)
            loss_file_pattern: Glob pattern for loss files (inferred if None)
            prediction_mode: Prediction strategy - 'sequential' or 'anchored'
                - sequential: Slide context window forward after each prediction
                - anchored: Keep context fixed, predict with increasing gaps
        """
        if prediction_mode not in ("sequential", "anchored"):
            raise ValueError(
                f"Invalid prediction_mode: {prediction_mode}. Must be 'sequential' or 'anchored'."
            )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.context_ratio = context_ratio
        self.use_real_data = use_real_data
        self.loss_file_pattern = loss_file_pattern
        self.prediction_mode = prediction_mode

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _evaluate_tasks(
        self,
        model_name: str,
        tasks: List[str],
        predict_fn,
        model_type: str,
    ) -> Tuple[Dict, List[Dict]]:
        """Common evaluation logic for all model types.

        Args:
            model_name: Name of the model to evaluate
            tasks: List of task names to evaluate
            predict_fn: Function that takes (steps, accuracies, ctx_end, num_future, task_name)
                       and returns predictions list
            model_type: Type of model for logging

        Returns:
            Tuple of (all_predictions dict, results list)
        """
        all_predictions = {}
        results = []

        for task_name in tasks:
            print(f"  Processing {task_name}...")

            try:
                # Load data with step alignment
                steps, accuracies, losses = load_step_aligned_data(
                    self.data_dir, model_name, task_name
                )

                ctx_end = int(len(steps) * self.context_ratio)
                future_accs = accuracies[ctx_end + 1 :]

                # Model predictions
                num_future = len(future_accs)
                model_preds = predict_fn(
                    steps, accuracies, ctx_end, num_future, task_name
                )
                model_errors = compute_errors(future_accs, model_preds)
                model_errors_per_step = compute_errors_per_timestep(
                    future_accs, model_preds
                )

                # Store predictions
                all_predictions[task_name] = {
                    "steps": steps.tolist(),
                    "accuracies": accuracies.tolist(),
                    "losses": losses.tolist(),
                    "context_end_idx": ctx_end,
                    "model_preds": model_preds,
                    "model_errors_per_step": model_errors_per_step.tolist(),
                }

                results.append(
                    {
                        "task": task_name,
                        "model_mae": model_errors["mae"],
                        "model_rmse": model_errors["rmse"],
                    }
                )

            except Exception as e:
                print(f"    Error: {e}")
                results.append(
                    {
                        "task": task_name,
                        "model_mae": float("nan"),
                        "model_rmse": float("nan"),
                    }
                )

        return all_predictions, results

    def _save_results(
        self,
        model_name: str,
        model_type: str,
        checkpoint: str,
        tasks: List[str],
        all_predictions: Dict,
        results: List[Dict],
    ):
        """Save evaluation results to files using standard format."""
        output_dir = Path(self.output_dir)

        # Convert results to TaskResult objects
        task_results = [
            TaskResult(
                task=r["task"],
                mae=r.get("model_mae", r.get("logistic_mae", float("nan"))),
                rmse=r.get("model_rmse", r.get("logistic_rmse", float("nan"))),
            )
            for r in results
        ]

        # Save results using EvalResults
        eval_results = EvalResults(
            model_name=model_name,
            model_type=model_type,
            tasks=tasks,
            results=task_results,
            checkpoint=checkpoint,
            context_ratio=self.context_ratio,
        )
        results_path = eval_results.save(output_dir)
        print(f"\nSaved results to {results_path}")

        # Convert predictions to TaskPredictions objects
        task_predictions = {}
        for task_name, preds in all_predictions.items():
            task_predictions[task_name] = TaskPredictions(
                steps=preds["steps"],
                accuracies=preds["accuracies"],
                losses=preds["losses"],
                predictions=preds.get("model_preds", preds.get("logistic_preds", [])),
                context_end_idx=preds.get("context_end_idx"),
                errors_per_step=preds.get(
                    "model_errors_per_step", preds.get("logistic_errors_per_step")
                ),
            )

        # Save predictions using EvalPredictions
        eval_predictions = EvalPredictions(
            model_name=model_name,
            model_type=model_type,
            task_predictions=task_predictions,
        )
        predictions_path = eval_predictions.save(output_dir)
        print(f"Saved predictions to {predictions_path}")

        return output_dir

    def baseline(
        self,
        model_name: str,
        checkpoint: str,
        tasks: List[str] = None,
    ):
        """Evaluate baseline model (transformer encoder only, no CNN).

        Args:
            model_name: Model name (e.g., DataDecide-c4-300M)
            checkpoint: Path to baseline model checkpoint
            tasks: Tasks to evaluate (auto-detected if None)
        """
        print(f"Using device: {self.device}")
        print(f"Loading baseline model from {checkpoint}...")
        model, config = load_model(checkpoint, self.device)
        print(f"  Parameters: {model.num_total_parameters():,}")

        # Determine loss file pattern
        loss_pattern = self.loss_file_pattern or "word_losses*.npy"
        print(f"Loss file pattern: {loss_pattern}")

        # Get soft prompt config from checkpoint (baseline doesn't use soft prompts, but keep consistent)
        soft_prompt_type = config["soft_prompt_type"]
        num_bins = config["num_bins"]
        # Get inverse perplexity config from checkpoint
        inverse_perplexity = config["inverse_perplexity"]
        bin_min = config["bin_min"]
        bin_max = config["bin_max"]

        print(f"Soft prompt type: {soft_prompt_type}")
        if inverse_perplexity:
            print(f"Inverse perplexity: enabled (bin_min={bin_min}, bin_max={bin_max})")

        # Get tasks
        available_tasks = get_available_tasks(self.data_dir, model_name)
        if tasks is None:
            tasks = available_tasks
        else:
            tasks = [t for t in tasks if t in available_tasks]
        print(f"Tasks to evaluate: {tasks}")

        print(f"\nPrediction mode: {self.prediction_mode}")
        if self.prediction_mode == "sequential":
            print(
                f"Evaluating with {'real' if self.use_real_data else 'predicted'} data for context"
            )

        def predict_fn(steps, accuracies, ctx_end, num_future, task_name):
            if self.prediction_mode == "anchored":
                return make_anchored_predictions(
                    model,
                    self.data_dir,
                    model_name,
                    steps,
                    accuracies,
                    ctx_end,  # anchor_idx
                    num_future,
                    self.device,
                    return_quantiles=True,
                    loss_file_pattern=loss_pattern,
                    soft_prompt_type=soft_prompt_type,
                    num_bins=num_bins,
                    inverse_perplexity=inverse_perplexity,
                    bin_min=bin_min,
                    bin_max=bin_max,
                )
            else:
                return make_sequential_predictions(
                    model,
                    self.data_dir,
                    model_name,
                    steps,
                    accuracies,
                    ctx_end,
                    num_future,
                    self.device,
                    use_real_data=self.use_real_data,
                    return_quantiles=True,
                    loss_file_pattern=loss_pattern,
                    soft_prompt_type=soft_prompt_type,
                    num_bins=num_bins,
                    inverse_perplexity=inverse_perplexity,
                    bin_min=bin_min,
                    bin_max=bin_max,
                )

        model_type_label = f"Baseline ({self.prediction_mode})"
        all_predictions, results = self._evaluate_tasks(
            model_name, tasks, predict_fn, model_type_label
        )

        model_type_save = (
            f"baseline_{self.prediction_mode}"
            if self.prediction_mode != "sequential"
            else "baseline"
        )
        output_dir = self._save_results(
            model_name, model_type_save, checkpoint, tasks, all_predictions, results
        )

        print_summary(results, model_type_label)
        print(f"\nEvaluation complete. Results saved to {output_dir}")

    def neural(
        self,
        model_name: str,
        checkpoint: str,
        tasks: List[str] = None,
    ):
        """Evaluate neural meta-model (with soft prompt generator + transformer encoder).

        Supports different soft prompt types:
        - cnn: Raw token losses processed by CNN (original architecture)
        - avg_loss: Average loss per step projected to embeddings

        Args:
            model_name: Model name (e.g., DataDecide-c4-300M)
            checkpoint: Path to neural model checkpoint
            tasks: Tasks to evaluate (auto-detected if None)
        """
        print(f"Using device: {self.device}")
        print(f"Loading neural model from {checkpoint}...")
        model, config = load_model(checkpoint, self.device)
        print(f"  Parameters: {model.num_total_parameters():,}")

        # Determine loss file pattern
        loss_pattern = self.loss_file_pattern or "word_losses*.npy"
        print(f"Loss file pattern: {loss_pattern}")

        # Get soft prompt config from checkpoint
        soft_prompt_type = config["soft_prompt_type"]
        num_bins = config["num_bins"]
        max_encoder_tokens = config.get("max_encoder_tokens")
        # Get inverse perplexity config from checkpoint
        inverse_perplexity = config["inverse_perplexity"]
        bin_min = config["bin_min"]
        bin_max = config["bin_max"]

        if inverse_perplexity:
            print(f"Inverse perplexity: enabled (bin_min={bin_min}, bin_max={bin_max})")
        if max_encoder_tokens:
            print(f"Max encoder tokens: {max_encoder_tokens}")

        # Get tasks
        available_tasks = get_available_tasks(self.data_dir, model_name)
        if tasks is None:
            tasks = available_tasks
        else:
            tasks = [t for t in tasks if t in available_tasks]
        print(f"Tasks to evaluate: {tasks}")

        print(f"\nPrediction mode: {self.prediction_mode}")
        if self.prediction_mode == "sequential":
            print(
                f"Evaluating with {'real' if self.use_real_data else 'predicted'} data for context"
            )

        def predict_fn(steps, accuracies, ctx_end, num_future, task_name):
            if self.prediction_mode == "anchored":
                return make_anchored_predictions(
                    model,
                    self.data_dir,
                    model_name,
                    steps,
                    accuracies,
                    ctx_end,  # anchor_idx
                    num_future,
                    self.device,
                    return_quantiles=True,
                    loss_file_pattern=loss_pattern,
                    soft_prompt_type=soft_prompt_type,
                    num_bins=num_bins,
                    inverse_perplexity=inverse_perplexity,
                    bin_min=bin_min,
                    bin_max=bin_max,
                    max_encoder_tokens=max_encoder_tokens,
                )
            else:
                return make_sequential_predictions(
                    model,
                    self.data_dir,
                    model_name,
                    steps,
                    accuracies,
                    ctx_end,
                    num_future,
                    self.device,
                    use_real_data=self.use_real_data,
                    return_quantiles=True,
                    loss_file_pattern=loss_pattern,
                    soft_prompt_type=soft_prompt_type,
                    num_bins=num_bins,
                    inverse_perplexity=inverse_perplexity,
                    bin_min=bin_min,
                    bin_max=bin_max,
                    max_encoder_tokens=max_encoder_tokens,
                )

        model_type_label = f"Neural ({self.prediction_mode})"
        all_predictions, results = self._evaluate_tasks(
            model_name, tasks, predict_fn, model_type_label
        )

        model_type_save = (
            f"neural_{self.prediction_mode}"
            if self.prediction_mode != "sequential"
            else "neural"
        )
        output_dir = self._save_results(
            model_name, model_type_save, checkpoint, tasks, all_predictions, results
        )

        print_summary(results, model_type_label)
        print(f"\nEvaluation complete. Results saved to {output_dir}")

    def probe(
        self,
        model_name: str,
        checkpoint: str,
        tasks: List[str] = None,
    ):
        """Evaluate probe model (CNN, histogram, or avg_loss probe) on ALL steps.

        Args:
            model_name: Model name (e.g., DataDecide-c4-300M)
            checkpoint: Path to probe model checkpoint
            tasks: Tasks to evaluate (auto-detected if None)
        """
        print("=" * 60)
        print("Probe Evaluation (All Steps)")
        print("=" * 60)
        print(f"Using device: {self.device}")
        print(f"Loading probe model from {checkpoint}...")

        # Peek at config to determine probe type and load appropriate model
        ckpt_peek = torch.load(checkpoint, map_location="cpu", weights_only=False)
        probe_type = ckpt_peek["config"]["probe_type"]

        if probe_type == "avg_loss":
            model, config, probe_tasks = load_avg_loss_probe_model(
                checkpoint, self.device
            )
        else:
            model, config, probe_tasks = load_probe_model(checkpoint, self.device)

        print(f"  Parameters: {model.num_parameters():,}")
        print(f"  Probe tasks: {probe_tasks}")

        is_histogram_probe = probe_type == "histogram"
        is_avg_loss_probe = probe_type == "avg_loss"
        print(f"  Probe type: {probe_type}")

        # Get evaluation directory
        model_dir = Path(self.data_dir) / model_name
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return

        # Determine loss file pattern from config or CLI
        if self.loss_file_pattern is not None:
            loss_pattern = self.loss_file_pattern
        elif "loss_file_pattern" in config:
            loss_pattern = config["loss_file_pattern"]
            print(f"  Using loss_file_pattern from probe config: {loss_pattern}")
        else:
            loss_pattern = "word_losses*.npy"
        print(f"Loss file pattern: {loss_pattern}")

        # Get inverse perplexity config from checkpoint
        inverse_perplexity = config["inverse_perplexity"]
        bin_min = config["bin_min"]
        bin_max = config["bin_max"]
        if inverse_perplexity:
            print(f"Inverse perplexity: enabled (bin_min={bin_min}, bin_max={bin_max})")

        # Load pre-computed histograms if this is a histogram probe
        hist_steps, histograms, hist_step_to_idx = None, None, None
        if is_histogram_probe:
            hist_steps, histograms, hist_step_to_idx = load_histograms_by_step(
                model_dir, inverse_perplexity=inverse_perplexity
            )
            if hist_steps is None:
                print(f"Error: No histograms_by_step.npz found for histogram probe")
                return
            print(f"  Loaded {len(hist_steps)} pre-computed histograms")

        # Load avg_losses_by_step.npy if this is an avg_loss probe
        step_to_avg_loss = {}
        if is_avg_loss_probe:
            avg_loss_file = model_dir / "avg_losses_by_step.npy"
            if not avg_loss_file.exists():
                print(f"Error: No avg_losses_by_step.npy found for avg_loss probe")
                return
            avg_loss_data = np.load(avg_loss_file)
            avg_loss_steps = avg_loss_data[0].astype(int)
            avg_loss_values = avg_loss_data[1].astype(np.float32)
            step_to_avg_loss = {
                int(s): float(v) for s, v in zip(avg_loss_steps, avg_loss_values)
            }
            print(f"  Loaded {len(step_to_avg_loss)} average loss values")

        # For CNN probe, find steps with loss files
        step_to_loss_file = {}
        if not is_histogram_probe and not is_avg_loss_probe:
            for item in model_dir.iterdir():
                if not item.is_dir():
                    continue
                step_num = _extract_step_number(item.name)
                if step_num is None:
                    continue
                loss_files = list(item.glob(loss_pattern))
                if loss_files:
                    step_to_loss_file[step_num] = loss_files[0]

            if len(step_to_loss_file) == 0:
                print(f"Error: No steps with loss files found for {model_name}")
                return

        # Get available steps based on probe type
        if is_histogram_probe:
            available_step_set = set(hist_step_to_idx.keys())
        elif is_avg_loss_probe:
            available_step_set = set(step_to_avg_loss.keys())
        else:
            available_step_set = set(step_to_loss_file.keys())

        # Load task accuracies using shared utility
        available_tasks = get_available_tasks(self.data_dir, model_name)
        if tasks is None:
            tasks = available_tasks
        else:
            tasks = [t for t in tasks if t in available_tasks]
        # Filter to tasks the probe was trained on
        eval_tasks = [t for t in tasks if t in probe_tasks]

        task_data, common_steps = load_task_accuracies(
            model_dir,
            eval_tasks,
            exclude_step0=True,
            required_steps=available_step_set,
        )

        if len(common_steps) == 0:
            print(f"Error: No common steps found for {model_name}")
            return

        print(f"Tasks to evaluate: {eval_tasks}")
        print(f"  Common steps: {len(common_steps)}")

        # Build step-to-accuracy mapping for each task
        task_step_to_acc = {}
        for task in task_data:
            task_step_to_acc[task] = {
                int(s): float(v)
                for s, v in zip(task_data[task]["steps"], task_data[task]["accuracies"])
            }

        # Check if this is a quantile model
        is_quantile = model.loss_type == "quantile"
        if is_quantile:
            quantiles = config["quantiles"]
            median_idx = (
                quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            )

        # Initialize per-task tracking
        task_predictions_per_step = {task: [] for task in eval_tasks}
        task_gt_per_step = {task: [] for task in eval_tasks}
        prediction_steps = []

        print(f"\nEvaluating probe predictions...")

        with torch.no_grad():
            for step in common_steps:
                prediction_steps.append(step)

                if is_histogram_probe:
                    # Use pre-computed histogram
                    histogram = histograms[hist_step_to_idx[step]]
                    input_t = torch.from_numpy(histogram).unsqueeze(0).to(self.device)
                elif is_avg_loss_probe:
                    # Use average loss value
                    avg_loss_val = step_to_avg_loss[step]
                    input_t = torch.tensor([avg_loss_val], dtype=torch.float32).to(
                        self.device
                    )
                else:
                    # Load raw losses for this step
                    raw_losses = np.load(step_to_loss_file[step]).flatten()
                    raw_losses = np.nan_to_num(
                        raw_losses, nan=0.0, posinf=10.0, neginf=0.0
                    )
                    raw_losses = np.clip(raw_losses, 0, 20)

                    # Truncate to expected input length (must match what model was trained with)
                    max_seq_len = config.get("max_seq_len")
                    if max_seq_len is not None and len(raw_losses) > max_seq_len:
                        raw_losses = raw_losses[:max_seq_len]

                    # Apply inverse perplexity transform if enabled: x -> e^(-x)
                    if inverse_perplexity:
                        raw_losses = np.exp(-raw_losses)
                        raw_losses = np.clip(raw_losses, bin_min, bin_max - 1e-6)

                    raw_losses = raw_losses.astype(np.float32)
                    input_t = torch.from_numpy(raw_losses).unsqueeze(0).to(self.device)

                # Single forward pass - output is (1, num_tasks) or (1, num_tasks, num_quantiles)
                output = model.forward(input_t)

                # Extract per-task predictions
                if is_quantile:
                    # output: (1, num_tasks, num_quantiles)
                    pred_values = output[0, :, median_idx].cpu().numpy()  # (num_tasks,)
                else:
                    # output: (1, num_tasks)
                    pred_values = output[0].cpu().numpy()  # (num_tasks,)

                # Store predictions for each task
                for task in eval_tasks:
                    task_idx = probe_tasks.index(task)
                    if step in task_step_to_acc[task]:
                        gt_acc = task_step_to_acc[task][step]
                        pred_acc = float(pred_values[task_idx])

                        task_predictions_per_step[task].append({"median": pred_acc})
                        task_gt_per_step[task].append(gt_acc)

        # Compute metrics and build results
        all_predictions = {}
        results = []

        for task in eval_tasks:
            task_preds = task_predictions_per_step[task]
            task_gt = task_gt_per_step[task]

            if len(task_preds) > 0:
                task_errors = [
                    abs(p["median"] - g) for p, g in zip(task_preds, task_gt)
                ]
                mae = np.mean(task_errors)
                rmse = np.sqrt(np.mean(np.array(task_errors) ** 2))

                all_predictions[task] = {
                    "steps": prediction_steps[: len(task_preds)],
                    "accuracies": task_gt,
                    "losses": [],  # Not stored per-step in this format
                    "model_preds": task_preds,
                    "model_errors_per_step": task_errors,
                }

                results.append(
                    {
                        "task": task,
                        "model_mae": mae,
                        "model_rmse": rmse,
                    }
                )

                print(f"  {task}: MAE={mae:.6f}, RMSE={rmse:.6f}")
            else:
                results.append(
                    {
                        "task": task,
                        "model_mae": float("nan"),
                        "model_rmse": float("nan"),
                    }
                )

        # Save results using shared method
        self._save_results(
            model_name, "probe", checkpoint, eval_tasks, all_predictions, results
        )

        print_summary(results, "Probe")
        print(f"\nEvaluation complete. Results saved to {self.output_dir}")

    def logistic(
        self,
        model_name: str,
        train_data_dir: str = "./results/datadecide_train",
        tasks: List[str] = None,
    ):
        """Evaluate zero-shot logistic baseline (fit on train, test on eval).

        Fits a 4-parameter logistic function on the training corpus and evaluates
        on ALL checkpoints of the eval model (no context required).

        Args:
            model_name: Model name to evaluate (e.g., DataDecide-c4-300M)
            train_data_dir: Directory containing training data
            tasks: Tasks to evaluate (auto-detected if None)
        """
        print("=" * 60)
        print("Zero-Shot Logistic Baseline")
        print("=" * 60)
        print(f"Training data: {train_data_dir}")
        print(f"Evaluation data: {self.data_dir}")
        print(f"Model: {model_name}")

        # Get tasks
        available_tasks = get_available_tasks(self.data_dir, model_name)
        if tasks is None:
            tasks = available_tasks
        else:
            tasks = [t for t in tasks if t in available_tasks]
        print(f"Tasks to evaluate: {len(tasks)} tasks")

        all_predictions = {}
        results = []

        for task_name in tasks:
            print(f"  Processing {task_name}...")

            try:
                # Load eval data with step alignment (ALL checkpoints, no context split)
                steps, accuracies, losses = load_step_aligned_data(
                    self.data_dir, model_name, task_name
                )

                # Fit logistic on training data and predict on ALL eval losses
                logistic_preds, params = make_logistic_prediction(
                    train_data_dir, losses, task_name
                )

                if params is None or len(logistic_preds) == 0:
                    print(f"    Warning: Logistic fit failed for {task_name}")
                    results.append(
                        {
                            "task": task_name,
                            "logistic_mae": float("nan"),
                            "logistic_rmse": float("nan"),
                            "num_train_points": 0,
                        }
                    )
                    continue

                # Compute errors against ALL eval checkpoints
                logistic_errors = compute_errors(accuracies, logistic_preds)
                logistic_errors_per_step = compute_errors_per_timestep(
                    accuracies, logistic_preds
                )

                # Count training points
                train_losses, _ = load_training_data_for_task(train_data_dir, task_name)
                num_train_points = len(train_losses)

                # Store predictions
                all_predictions[task_name] = {
                    "steps": steps.tolist(),
                    "accuracies": accuracies.tolist(),
                    "losses": losses.tolist(),
                    "logistic_preds": logistic_preds,
                    "logistic_params": params.tolist() if params is not None else None,
                    "logistic_errors_per_step": logistic_errors_per_step.tolist(),
                    "num_train_points": num_train_points,
                }

                results.append(
                    {
                        "task": task_name,
                        "logistic_mae": logistic_errors["mae"],
                        "logistic_rmse": logistic_errors["rmse"],
                        "num_train_points": num_train_points,
                    }
                )

            except Exception as e:
                print(f"    Error: {e}")
                import traceback

                traceback.print_exc()
                results.append(
                    {
                        "task": task_name,
                        "logistic_mae": float("nan"),
                        "logistic_rmse": float("nan"),
                        "num_train_points": 0,
                    }
                )

        # Save results using standard format
        output_dir = Path(self.output_dir) / "logistic"

        # Convert results to TaskResult objects
        task_results = [
            TaskResult(
                task=r["task"],
                mae=r["logistic_mae"],
                rmse=r["logistic_rmse"],
                num_samples=r.get("num_train_points"),
            )
            for r in results
        ]

        # Save results using EvalResults
        eval_results = EvalResults(
            model_name=model_name,
            model_type="logistic",
            tasks=tasks,
            results=task_results,
            config={"train_data_dir": train_data_dir},
        )
        results_path = eval_results.save(output_dir)
        print(f"\nSaved results to {results_path}")

        # Convert predictions to TaskPredictions objects
        task_predictions = {}
        for task_name, preds in all_predictions.items():
            task_predictions[task_name] = TaskPredictions(
                steps=preds["steps"],
                accuracies=preds["accuracies"],
                losses=preds["losses"],
                predictions=preds.get("logistic_preds", []),
                errors_per_step=preds.get("logistic_errors_per_step"),
            )

        # Save predictions using EvalPredictions
        eval_predictions = EvalPredictions(
            model_name=model_name,
            model_type="logistic",
            task_predictions=task_predictions,
        )
        predictions_path = eval_predictions.save(output_dir)
        print(f"Saved predictions to {predictions_path}")

        # Print summary
        print("\n" + "=" * 48)
        print("SUMMARY (Zero-Shot Logistic): Mean Absolute Error")
        print("=" * 48)
        print(f"{'Task':<25} {'MAE':>10} {'Train Pts':>12}")
        print("-" * 48)
        for r in results:
            print(
                f"{r['task']:<25} {r['logistic_mae']:>10.4f} {r['num_train_points']:>12}"
            )
        print("-" * 48)

        mean_mae = np.nanmean([r["logistic_mae"] for r in results])
        total_train = sum(r["num_train_points"] for r in results)
        print(f"{'Mean':<25} {mean_mae:>10.4f} {total_train:>12}")
        print("=" * 48)

        print(f"\nEvaluation complete. Results saved to {output_dir}")

    def delta_probe(
        self,
        model_name: str,
        checkpoint: str,
        tasks: List[str] = None,
        anchor_step_idx: int = 0,
        delta_eval_mode: str = "anchor",
    ):
        """Evaluate delta probe model using anchored or cumulative trajectory prediction.

        Two evaluation modes:
        1. 'anchor' mode: Given a starting point (X1, Y1), predicts all future accuracies as:
            Y_t = Y1 + model(X_t - X1)
           This takes larger and larger gaps from anchor.

        2. 'cumulative' mode: Predicts step-by-step with gap-1, accumulating predictions:
            Y_{t+1} = Y_t + model(X_{t+1} - X_t)
           This always uses gap-1 deltas and accumulates the predicted Y values.

        Args:
            model_name: Model name (e.g., DataDecide-c4-300M)
            checkpoint: Path to delta probe model checkpoint
            tasks: Tasks to evaluate (auto-detected if None)
            anchor_step_idx: Index of anchor step (0 = first available step)
            delta_eval_mode: Evaluation mode - 'anchor' or 'cumulative'
        """
        if delta_eval_mode not in ("anchor", "cumulative"):
            raise ValueError(
                f"Invalid delta_eval_mode: {delta_eval_mode}. Must be 'anchor' or 'cumulative'."
            )
        mode_desc = (
            "Anchored Trajectory"
            if delta_eval_mode == "anchor"
            else "Cumulative (Gap-1)"
        )
        print("=" * 60)
        print(f"Delta Probe Evaluation ({mode_desc})")
        print("=" * 60)
        print(f"Evaluation mode: {delta_eval_mode}")
        print(f"Using device: {self.device}")
        print(f"Loading delta probe model from {checkpoint}...")

        model, config, probe_tasks = load_delta_probe_model(checkpoint, self.device)
        probe_type = config["probe_type"]
        num_bins = config["num_bins"]
        bin_min = config["bin_min"]
        bin_max = config["bin_max"]
        # Get inverse perplexity config from checkpoint
        inverse_perplexity = config["inverse_perplexity"]

        print(f"  Probe type: {probe_type}")
        print(f"  Parameters: {model.num_parameters():,}")
        print(f"  Num bins: {num_bins}")
        if inverse_perplexity:
            print(
                f"  Inverse perplexity: enabled (bin_min={bin_min}, bin_max={bin_max})"
            )

        # Get evaluation directory
        model_dir = Path(self.data_dir) / model_name
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return

        # Load pre-computed histograms
        hist_steps, histograms, hist_step_to_idx = load_histograms_by_step(
            model_dir, inverse_perplexity=inverse_perplexity
        )
        if hist_steps is None:
            print(f"Error: No histograms_by_step.npz found for {model_name}")
            return
        print(f"  Loaded {len(hist_steps)} pre-computed histograms")

        # Get available tasks
        available_tasks = get_available_tasks(self.data_dir, model_name)
        if tasks is None:
            tasks = available_tasks
        else:
            tasks = [t for t in tasks if t in available_tasks]

        # Filter to tasks the probe was trained on
        eval_tasks = [t for t in probe_tasks if t in tasks]
        print(f"Tasks to evaluate: {len(eval_tasks)} tasks")

        # Load task accuracies using shared utility
        task_data, common_steps = load_task_accuracies(
            model_dir,
            eval_tasks,
            exclude_step0=True,
            required_steps=set(hist_step_to_idx.keys()),
        )

        if len(common_steps) < 2:
            print(f"Error: Not enough common steps for {model_name}")
            return

        # Validate anchor index
        if anchor_step_idx >= len(common_steps):
            print(
                f"Warning: Anchor index {anchor_step_idx} >= num steps {len(common_steps)}, using 0"
            )
            anchor_step_idx = 0

        anchor_step = common_steps[anchor_step_idx]
        print(f"Anchor step: {anchor_step} (index {anchor_step_idx})")
        print(f"  Common steps: {len(common_steps)}")

        # Build step-to-accuracy mapping for each task
        task_step_to_acc = {}
        for task in task_data:
            task_step_to_acc[task] = {
                int(s): float(v)
                for s, v in zip(task_data[task]["steps"], task_data[task]["accuracies"])
            }

        # Get anchor histogram from pre-computed histograms
        anchor_hist = histograms[hist_step_to_idx[anchor_step]]

        print(f"  Probe tasks: {len(probe_tasks)}")
        print(f"  Eval tasks: {len(eval_tasks)}")

        # Evaluate: predict using anchor or cumulative mode
        # Model outputs (batch, num_tasks) for MSE or (batch, num_tasks, num_quantiles) for quantile
        is_kl_probe = isinstance(model, KLDeltaProbe)
        is_quantile = config["loss_type"] == "quantile"
        if is_quantile:
            quantiles = config["quantiles"]
            median_idx = (
                quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            )

        # Initialize per-task tracking
        task_predictions_per_step = {task: [] for task in eval_tasks}
        task_gt_per_step = {task: [] for task in eval_tasks}
        prediction_steps = []

        # For cumulative mode, track the accumulated prediction per task
        if delta_eval_mode == "cumulative":
            task_accumulated_pred = {task: None for task in eval_tasks}

        print(f"\nEvaluating {delta_eval_mode} trajectory prediction...")

        with torch.no_grad():
            for i, step in enumerate(common_steps):
                if i <= anchor_step_idx:
                    continue  # Skip anchor and earlier steps

                prediction_steps.append(step)
                prev_step = common_steps[i - 1]

                # Get histograms from pre-computed data
                hist_curr = histograms[hist_step_to_idx[step]]
                hist_prev = histograms[hist_step_to_idx[prev_step]]

                if delta_eval_mode == "anchor":
                    # Anchor mode: predict delta from anchor to current step
                    # Y_t = Y_anchor + model(X_t - X_anchor)
                    if is_kl_probe:
                        hist_curr_tensor = (
                            torch.from_numpy(hist_curr).unsqueeze(0).to(self.device)
                        )
                        anchor_hist_tensor = (
                            torch.from_numpy(anchor_hist).unsqueeze(0).to(self.device)
                        )
                        pred_delta = model.forward(hist_curr_tensor, anchor_hist_tensor)
                    else:
                        delta_hist = hist_curr - anchor_hist
                        delta_hist_tensor = (
                            torch.from_numpy(delta_hist).unsqueeze(0).to(self.device)
                        )
                        pred_delta = model.forward(delta_hist_tensor)
                else:
                    # Cumulative mode: predict delta from previous step to current step (gap-1)
                    # Y_{t+1} = Y_t + model(X_{t+1} - X_t)
                    if is_kl_probe:
                        hist_curr_tensor = (
                            torch.from_numpy(hist_curr).unsqueeze(0).to(self.device)
                        )
                        hist_prev_tensor = (
                            torch.from_numpy(hist_prev).unsqueeze(0).to(self.device)
                        )
                        pred_delta = model.forward(hist_curr_tensor, hist_prev_tensor)
                    else:
                        delta_hist = hist_curr - hist_prev
                        delta_hist_tensor = (
                            torch.from_numpy(delta_hist).unsqueeze(0).to(self.device)
                        )
                        pred_delta = model.forward(delta_hist_tensor)

                # Extract per-task predictions
                if is_quantile:
                    # pred_delta: (1, num_tasks, num_quantiles)
                    pred_delta_median = (
                        pred_delta[0, :, median_idx].cpu().numpy()
                    )  # (num_tasks,)
                else:
                    # pred_delta: (1, num_tasks)
                    pred_delta_median = pred_delta[0].cpu().numpy()  # (num_tasks,)

                # Store predictions for each task
                for task in eval_tasks:
                    task_idx = probe_tasks.index(task)
                    if (
                        anchor_step in task_step_to_acc[task]
                        and step in task_step_to_acc[task]
                    ):
                        gt_acc = task_step_to_acc[task][step]

                        if delta_eval_mode == "anchor":
                            # Anchor mode: add delta to anchor accuracy
                            anchor_acc = task_step_to_acc[task][anchor_step]
                            pred_acc = anchor_acc + pred_delta_median[task_idx]
                        else:
                            # Cumulative mode: add delta to accumulated prediction
                            if task_accumulated_pred[task] is None:
                                # First prediction: start from anchor ground truth
                                task_accumulated_pred[task] = task_step_to_acc[task][
                                    anchor_step
                                ]
                            pred_acc = (
                                task_accumulated_pred[task]
                                + pred_delta_median[task_idx]
                            )
                            task_accumulated_pred[task] = (
                                pred_acc  # Update accumulated value
                            )

                        task_predictions_per_step[task].append(
                            {"median": float(pred_acc)}
                        )
                        task_gt_per_step[task].append(gt_acc)

        # Compute metrics and build results
        all_predictions = {}
        results = []

        for task in eval_tasks:
            task_preds = task_predictions_per_step[task]
            task_gt = task_gt_per_step[task]

            if len(task_preds) > 0:
                task_errors = [
                    abs(p["median"] - g) for p, g in zip(task_preds, task_gt)
                ]
                mae = np.mean(task_errors)
                rmse = np.sqrt(np.mean(np.array(task_errors) ** 2))

                # Get full trajectory including anchor for visualization
                full_steps = [anchor_step] + prediction_steps[: len(task_preds)]
                full_accs = [task_step_to_acc[task][anchor_step]] + task_gt
                full_preds = [
                    {"median": task_step_to_acc[task][anchor_step]}
                ] + task_preds

                all_predictions[task] = {
                    "steps": full_steps,
                    "accuracies": full_accs,
                    "losses": [],  # Delta probe doesn't use losses directly
                    "model_preds": full_preds,
                    "model_errors_per_step": [0.0] + task_errors,  # 0 error at anchor
                }

                results.append(
                    {
                        "task": task,
                        "model_mae": mae,
                        "model_rmse": rmse,
                        "num_samples": len(task_preds),
                    }
                )

                print(f"  {task}: MAE={mae:.6f}, RMSE={rmse:.6f}")

        # Save results using shared method
        self._save_results(
            model_name, "delta_probe", checkpoint, eval_tasks, all_predictions, results
        )

        # Print summary
        mode_summary = "Anchored" if delta_eval_mode == "anchor" else "Cumulative"
        print("\n" + "=" * 48)
        print(f"SUMMARY (Delta Probe - {mode_summary})")
        print("=" * 48)
        print(f"{'Task':<25} {'MAE':>10} {'Samples':>10}")
        print("-" * 48)
        for r in results:
            print(f"{r['task']:<25} {r['model_mae']:>10.6f} {r['num_samples']:>10}")
        print("-" * 48)

        mean_mae = np.nanmean([r["model_mae"] for r in results])
        total_samples = sum(r["num_samples"] for r in results)
        print(f"{'Mean':<25} {mean_mae:>10.6f} {total_samples:>10}")
        print("=" * 48)

        print(f"\nEvaluation complete. Results saved to {self.output_dir}")


def main():
    fire.Fire(ScalingEvaluator)


if __name__ == "__main__":
    main()
