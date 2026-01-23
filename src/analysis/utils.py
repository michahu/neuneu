"""Shared utility functions for analysis and visualization."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

from src.meta.utils import _extract_step_number


def load_histograms_by_step(
    model_dir: Path,
    file_name: str = "histograms_by_step.npz",
    inverse_perplexity: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[int, int]]]:
    """
    Load pre-computed histograms from histograms_by_step.npz.

    Args:
        model_dir: Path to model directory containing histograms_by_step.npz

    Returns:
        Tuple of:
        - steps: Array of step numbers (int64)
        - histograms: Array of histograms (num_steps, num_bins), float32
        - step_to_idx: Mapping from step number to histogram index
        Returns (None, None, None) if file doesn't exist.
    """
    model_dir = Path(model_dir)

    # Append _invp to filename if using inverse perplexity transform
    if inverse_perplexity:
        base, ext = os.path.splitext(file_name)
        file_name = f"{base}_invp{ext}"

    hist_file = model_dir / file_name
    if not hist_file.exists():
        return None, None, None

    hist_data = np.load(hist_file)
    steps = hist_data["steps"].astype(np.int64)
    histograms = hist_data["histograms"].astype(np.float32)
    step_to_idx = {int(s): i for i, s in enumerate(steps)}

    return steps, histograms, step_to_idx


def load_step_aligned_data(
    data_dir: str,
    model_name: str,
    task_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load accuracy and loss data aligned by step.

    Task files and loss files may have different steps. This function finds
    the common steps and returns aligned arrays.

    Args:
        data_dir: Directory containing model data
        model_name: Model name (e.g., "DataDecide-c4-300M")
        task_name: Task name (e.g., "arc_challenge")

    Returns:
        Tuple of (steps, accuracies, losses) arrays, all aligned by step
    """
    model_dir = Path(data_dir) / model_name

    # Load task data (steps, accuracies)
    task_file = model_dir / f"{task_name}.npy"
    task_data = np.load(task_file)
    task_steps = task_data[0].astype(int)
    task_accs = task_data[1]

    # Load loss data (steps, losses)
    loss_file = model_dir / "avg_losses_by_step.npy"
    loss_data = np.load(loss_file)
    loss_steps = loss_data[0].astype(int)
    loss_values = loss_data[1]

    # Build step-to-value mappings
    task_step_to_acc = {int(s): a for s, a in zip(task_steps, task_accs)}
    loss_step_to_val = {int(s): l for s, l in zip(loss_steps, loss_values)}

    # Find common steps (preserving task step order), excluding step 0
    common_steps = [s for s in task_steps if s in loss_step_to_val and s != 0]

    # Build aligned arrays
    steps = np.array(common_steps, dtype=int)
    accuracies = np.array([task_step_to_acc[s] for s in common_steps])
    losses = np.array([loss_step_to_val[s] for s in common_steps])

    return steps, accuracies, losses


def find_common_steps(
    *step_arrays: np.ndarray,
    preserve_order_from: int = 0,
) -> List[int]:
    """
    Find steps common to all provided step arrays.

    Args:
        *step_arrays: Variable number of step arrays to intersect
        preserve_order_from: Index of array whose order to preserve (default: first)

    Returns:
        List of common steps in the order of the specified array
    """
    if not step_arrays:
        return []

    # Convert all to sets for intersection
    step_sets = [set(arr.astype(int)) for arr in step_arrays]
    common = step_sets[0]
    for s in step_sets[1:]:
        common &= s

    # Preserve order from specified array
    order_array = step_arrays[preserve_order_from]
    return [int(s) for s in order_array if int(s) in common]


def extract_model_size(model_name: str) -> str:
    """
    Extract model size from model name like 'DataDecide-c4-90M'.

    Args:
        model_name: Full model name

    Returns:
        Size string like '90M', '1B', etc.
    """
    parts = model_name.split("-")
    for part in parts:
        if part.endswith("M") or part.endswith("B"):
            return part
    return model_name  # Fallback


def logistic_function(
    L: np.ndarray, a: float, k: float, L0: float, b: float
) -> np.ndarray:
    """
    4-parameter logistic function for accuracy vs loss.

    Args:
        L: Input values (typically log loss)
        a: Amplitude (max - min)
        k: Steepness
        L0: Midpoint
        b: Offset (minimum value)

    Returns:
        Predicted accuracy values
    """
    return a / (1 + np.exp(-k * (L - L0))) + b


def fit_logistic_scaling_law(
    losses: np.ndarray,
    accuracies: np.ndarray,
    use_log_loss: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit a logistic scaling law to predict accuracy from loss.

    Args:
        losses: Array of loss values
        accuracies: Array of accuracy values
        use_log_loss: Whether to use log(loss) as input

    Returns:
        Tuple of (parameters, covariance) or (None, None) if fit fails
    """
    if use_log_loss:
        L = np.log(losses + 1e-8)
    else:
        L = losses

    initial_guess = [
        np.max(accuracies) - np.min(accuracies),
        -1.0,
        np.median(L),
        np.min(accuracies),
    ]

    try:
        popt, pcov = curve_fit(
            logistic_function,
            L,
            accuracies,
            p0=initial_guess,
            maxfev=10000,
            bounds=([-2, -100, -10, 0], [2, 100, 10, 1]),
        )
        return popt, pcov
    except Exception as e:
        print(f"Warning: Logistic fit failed: {e}")
        return None, None


def predict_logistic(
    losses: np.ndarray,
    params: np.ndarray,
    use_log_loss: bool = True,
) -> np.ndarray:
    """
    Predict accuracy using fitted logistic parameters.

    Args:
        losses: Array of loss values to predict for
        params: Fitted logistic parameters (a, k, L0, b)
        use_log_loss: Whether to use log(loss) as input

    Returns:
        Predicted accuracy values
    """
    if use_log_loss:
        L = np.log(losses + 1e-8)
    else:
        L = losses
    return logistic_function(L, *params)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics between predictions and targets.

    Args:
        predictions: Array of predicted values
        targets: Array of target values

    Returns:
        Dictionary with mae, mse, rmse, and correlation
    """
    errors = targets - predictions
    mae = np.abs(errors).mean()
    mse = (errors**2).mean()
    rmse = np.sqrt(mse)

    # Correlation
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    correlation = (pred_centered * target_centered).sum() / (
        np.linalg.norm(pred_centered) * np.linalg.norm(target_centered) + 1e-8
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "correlation": correlation,
    }


def compute_errors(
    ground_truth: np.ndarray,
    predictions: List[Dict],
    key: str = "median",
) -> Dict[str, float]:
    """
    Compute MAE and RMSE for predictions stored as list of dicts.

    Args:
        ground_truth: Array of ground truth values
        predictions: List of prediction dicts with the specified key
        key: Key to use for prediction values (e.g., "median")

    Returns:
        Dictionary with mae and rmse
    """
    pred_values = np.array([p[key] for p in predictions])
    n = min(len(ground_truth), len(pred_values))

    errors = ground_truth[:n] - pred_values[:n]
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())

    return {"mae": mae, "rmse": rmse}


def compute_errors_per_timestep(
    ground_truth: np.ndarray,
    predictions: List[Dict],
    key: str = "median",
) -> np.ndarray:
    """
    Compute absolute error at each timestep.

    Args:
        ground_truth: Array of ground truth values
        predictions: List of prediction dicts
        key: Key to use for prediction values

    Returns:
        Array of absolute errors per timestep
    """
    pred_values = np.array([p[key] for p in predictions])
    n = min(len(ground_truth), len(pred_values))
    return np.abs(ground_truth[:n] - pred_values[:n])


def create_multi_task_grid(
    tasks: List[str],
    plot_fn,
    title: str = "Multi-Task Visualization",
    output_path: Optional[str] = None,
    n_cols: int = 4,
    subplot_size: Tuple[float, float] = (5.0, 4.0),
):
    """
    Create a grid of subplots for multiple tasks.

    Args:
        tasks: List of task names
        plot_fn: Function that takes (ax, task_name, task_idx) and plots to the axis
        title: Overall figure title
        output_path: Path to save the figure
        n_cols: Number of columns in the grid
        subplot_size: (width, height) for each subplot

    Returns:
        Figure object
    """
    n_tasks = len(tasks)
    n_cols = min(n_cols, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(subplot_size[0] * n_cols, subplot_size[1] * n_rows)
    )

    # Normalize axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, task_name in enumerate(tasks):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        try:
            plot_fn(ax, task_name, idx)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # Hide empty subplots
    for idx in range(n_tasks, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
    return fig


# ==============================================================================
# Model Loading Functions
# ==============================================================================


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple of (model, config dict)
    """
    from src.meta.model import MetaLossPredictor, MetaLossPredictorBaseline

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    base_kwargs = dict(
        loss_type=config["loss_type"],
        quantiles=config.get("quantiles"),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
    )

    encoder_kwargs = dict(
        soft_prompt_type=config["soft_prompt_type"],
        num_bins=config["num_bins"],
        cnn_input_len=config["max_encoder_tokens"],
        cnn_channels=tuple(config["cnn_channels"]),
        cnn_kernel_size=config["cnn_kernel_size"],
        cnn_stride=config["cnn_stride"],
    )

    if config.get("use_baseline", False):
        baseline_kwargs = {k: v for k, v in base_kwargs.items()}
        model = MetaLossPredictorBaseline(**baseline_kwargs)
    else:
        model = MetaLossPredictor(**base_kwargs, **encoder_kwargs)

    strict = not config.get("use_baseline", False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model = model.to(device)
    model.eval()
    return model, config


def load_probe_model(checkpoint_path: str, device: torch.device):
    """Load probe model from checkpoint (CNN or histogram).

    Args:
        checkpoint_path: Path to probe model checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple of (model, config dict, list of tasks)
    """
    from src.meta.probes import DirectProbe, HistogramProbe

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    quantiles = config["quantiles"]
    if quantiles is not None:
        quantiles = list(quantiles)

    probe_type = config["probe_type"]

    # Tasks are stored separately in checkpoint (not inside config) since config.tasks may be None
    tasks = checkpoint.get("tasks") or config["tasks"]
    num_tasks = len(tasks)

    if probe_type == "histogram":
        model = HistogramProbe(
            num_bins=config["num_bins"],
            hidden_dims=tuple(config["hidden_dims"]),
            loss_type=config["loss_type"],
            num_tasks=num_tasks,
            quantiles=quantiles,
            use_rms_norm=config["use_rms_norm"],
        )
    else:
        model = DirectProbe(
            loss_type=config["loss_type"],
            num_tasks=num_tasks,
            quantiles=quantiles,
            input_len=config["max_seq_len"],
            channels=tuple(config["channels"]),
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            encoder_hidden_dim=config.get("encoder_hidden_dim", 512),
            hidden_dim=config["hidden_dim"],
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, tasks


def load_avg_loss_probe_model(checkpoint_path: str, device: torch.device):
    """Load avg_loss probe model from checkpoint.

    Args:
        checkpoint_path: Path to avg_loss probe model checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple of (model, config dict, list of tasks)
    """
    from src.meta.probes import AverageLossProbe

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    quantiles = config["quantiles"]
    if quantiles is not None:
        quantiles = list(quantiles)

    # Tasks are stored separately in checkpoint (not inside config) since config.tasks may be None
    tasks = checkpoint.get("tasks") or config["tasks"]
    num_tasks = len(tasks)

    model = AverageLossProbe(
        hidden_dims=tuple(config["hidden_dims"]),
        loss_type=config["loss_type"],
        num_tasks=num_tasks,
        quantiles=quantiles,
        inverse_perplexity=config["inverse_perplexity"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, tasks


def load_delta_probe_model(checkpoint_path: str, device: torch.device):
    """Load delta probe model from checkpoint (delta or kl_delta).

    Args:
        checkpoint_path: Path to delta probe model checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple of (model, config dict, list of tasks)
    """
    from src.meta.probes import DeltaHistogramProbe, KLDeltaProbe

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    probe_type = config["probe_type"]

    quantiles = config["quantiles"]
    if quantiles is not None:
        quantiles = list(quantiles)

    # Tasks are stored separately in checkpoint (not inside config) since config.tasks may be None
    tasks = checkpoint.get("tasks") or config["tasks"]
    num_tasks = len(tasks)

    if probe_type == "kl_delta":
        model = KLDeltaProbe(
            num_bins=config["num_bins"],
            hidden_dims=tuple(config["hidden_dims"]),
            loss_type=config["loss_type"],
            quantiles=quantiles,
            num_tasks=num_tasks,
        )
    else:
        model = DeltaHistogramProbe(
            num_bins=config["num_bins"],
            hidden_dims=tuple(config["hidden_dims"]),
            loss_type=config["loss_type"],
            quantiles=quantiles,
            num_tasks=num_tasks,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, tasks


# ==============================================================================
# Data Loading Functions
# ==============================================================================


def load_accuracy_data(
    data_dir: str, model_name: str, task_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load accuracy data for a specific model and task.

    Args:
        data_dir: Base data directory
        model_name: Name of the model (e.g., "DataDecide-c4-300M")
        task_name: Name of the task (e.g., "arc_challenge")

    Returns:
        Tuple of (steps array, accuracies array)
    """
    task_file = Path(data_dir) / model_name / f"{task_name}.npy"
    data = np.load(task_file)
    steps = data[0].astype(int)
    accuracies = data[1]
    return steps, accuracies


def load_loss_data(data_dir: str, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load average loss data for a model.

    Args:
        data_dir: Base data directory
        model_name: Name of the model (e.g., "DataDecide-c4-300M")

    Returns:
        Tuple of (steps array, losses array)
    """
    loss_file = Path(data_dir) / model_name / "avg_losses_by_step.npy"
    data = np.load(loss_file)
    steps = data[0].astype(int)
    losses = data[1]
    return steps, losses


def load_raw_losses(
    data_dir: str,
    model_name: str,
    step: int,
    max_tokens: int = 1000000,
    loss_file_pattern: str = "word_losses*.npy",
    inverse_perplexity: bool = False,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
) -> Optional[np.ndarray]:
    """Load raw token-level losses for a specific step.

    Args:
        data_dir: Base data directory
        model_name: Name of the model (e.g., "DataDecide-c4-300M")
        step: Training step number
        max_tokens: Maximum number of tokens to load
        loss_file_pattern: Glob pattern for loss files
        inverse_perplexity: If True, transform losses x -> e^(-x) (values become 0-1)
        bin_min: Minimum value for clipping (0.0 for inverse_perplexity)
        bin_max: Maximum value for clipping (1.0 for inverse_perplexity, 15.0 otherwise)

    Returns:
        Array of raw losses, or None if not found
    """
    model_dir = Path(data_dir) / model_name
    for item in model_dir.iterdir():
        if item.is_dir():
            extracted = _extract_step_number(item.name)
            if extracted == step:
                npy_files = list(item.glob(loss_file_pattern))
                if npy_files:
                    losses = np.load(npy_files[0]).flatten()
                    losses = np.nan_to_num(losses, nan=0.0, posinf=10.0, neginf=0.0)
                    losses = np.clip(losses, 0, 20)

                    # Apply inverse perplexity transform if enabled: x -> e^(-x)
                    if inverse_perplexity:
                        losses = np.exp(-losses)
                        losses = np.clip(losses, bin_min, bin_max - 1e-6)

                    if len(losses) > max_tokens:
                        losses = losses[:max_tokens]
                    return losses.astype(np.float32)
    return None


def load_training_data_for_task(
    train_data_dir: str,
    task_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all (loss, accuracy) pairs for a task from training corpus.

    Aggregates data across all models in the training directory.

    Args:
        train_data_dir: Path to training data directory (e.g., ./results/datadecide_train)
        task_name: Name of the task to load

    Returns:
        Tuple of (losses array, accuracies array) aggregated across all models
    """
    all_losses = []
    all_accuracies = []

    train_dir = Path(train_data_dir)
    for model_dir in train_dir.iterdir():
        if not model_dir.is_dir():
            continue

        loss_file = model_dir / "avg_losses_by_step.npy"
        task_file = model_dir / f"{task_name}.npy"

        if not loss_file.exists() or not task_file.exists():
            continue

        loss_data = np.load(loss_file)
        task_data = np.load(task_file)

        # Align by step
        loss_steps = set(loss_data[0].astype(int))
        task_steps = set(task_data[0].astype(int))
        common_steps = sorted(loss_steps & task_steps)

        loss_step_to_idx = {int(s): i for i, s in enumerate(loss_data[0])}
        task_step_to_idx = {int(s): i for i, s in enumerate(task_data[0])}

        for step in common_steps:
            if step == 0:
                continue  # Skip step 0
            all_losses.append(loss_data[1][loss_step_to_idx[step]])
            all_accuracies.append(task_data[1][task_step_to_idx[step]])

    return np.array(all_losses), np.array(all_accuracies)


def get_available_tasks(data_dir: str, model_name: str) -> List[str]:
    """Get list of available tasks for a model.

    Args:
        data_dir: Base data directory
        model_name: Name of the model

    Returns:
        List of task names (file stems excluding avg_losses_by_step)
    """
    model_dir = Path(data_dir) / model_name
    return [f.stem for f in model_dir.glob("*.npy") if f.stem != "avg_losses_by_step"]


# ==============================================================================
# Utility Functions
# ==============================================================================


def compute_histogram(
    losses: np.ndarray,
    num_bins: int = 32,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    inverse_perplexity: bool = False,
) -> np.ndarray:
    """Compute normalized histogram from loss values.

    Args:
        losses: Array of loss values
        num_bins: Number of histogram bins
        bin_min: Minimum value for binning
        bin_max: Maximum value for binning
        inverse_perplexity: If True, transform losses x -> e^(-x) before binning

    Returns:
        Normalized histogram array of shape (num_bins,)
    """
    losses = losses.flatten()
    losses = np.nan_to_num(losses, nan=0.0, posinf=10.0, neginf=0.0)

    # Apply inverse perplexity transform if enabled: x -> e^(-x)
    if inverse_perplexity:
        losses = np.exp(
            -np.clip(losses, 0, 20)
        )  # Clip before transform to avoid overflow

    losses = np.clip(losses, bin_min, bin_max - 1e-6)
    hist, _ = np.histogram(losses, bins=num_bins, range=(bin_min, bin_max))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def compute_gaps(indices: np.ndarray, target_idx: int) -> np.ndarray:
    """Compute forward gaps from indices to target.

    Args:
        indices: Array of context indices
        target_idx: Target index to compute gaps to

    Returns:
        Array of gap values
    """
    if len(indices) == 0:
        return np.array([], dtype=np.int64)

    gaps = np.zeros(len(indices), dtype=np.int64)
    gaps[0] = indices[0] + 1
    if len(indices) > 1:
        gaps[1:-1] = np.diff(indices)[:-1]
    gaps[-1] = target_idx - indices[-1]
    return gaps


def print_summary(results: List[Dict], model_type: str):
    """Print summary table of results.

    Args:
        results: List of result dicts with 'task', 'model_mae', 'model_rmse' keys
        model_type: Name of the model type for display
    """
    print("\n" + "=" * 48)
    print(f"SUMMARY ({model_type}): Mean Absolute Error")
    print("=" * 48)
    print(f"{'Task':<20} {'MAE':>12} {'RMSE':>12}")
    print("-" * 48)
    for r in results:
        model_mae = r.get("model_mae", float("nan"))
        model_rmse = r.get("model_rmse", float("nan"))
        print(f"{r['task']:<20} {model_mae:>12.4f} {model_rmse:>12.4f}")
    print("-" * 48)

    model_mean = np.nanmean([r.get("model_mae", float("nan")) for r in results])
    rmse_mean = np.nanmean([r.get("model_rmse", float("nan")) for r in results])
    print(f"{'Mean':<20} {model_mean:>12.4f} {rmse_mean:>12.4f}")
    print("=" * 48)
