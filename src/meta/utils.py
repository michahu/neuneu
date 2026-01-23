"""
Utility functions for the meta-loss predictor.
"""

import os
import re
import logging
from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: torch.Tensor | List[float],
) -> torch.Tensor:
    """
    Pinball loss for quantile regression.

    Args:
        predictions: (..., num_quantiles) predicted values for each quantile
        targets: (...,) actual values
        quantiles: (num_quantiles,) quantile levels (e.g., [0.1, 0.5, 0.9])
            Can be a tensor or list of floats.

    Returns:
        Scalar pinball loss (mean over all elements and quantiles)
    """
    # Convert quantiles to tensor if needed
    if isinstance(quantiles, list):
        quantiles = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)

    targets = targets.unsqueeze(-1)  # (..., 1)
    errors = targets - predictions  # (..., num_quantiles)

    quantiles = quantiles.view(1, -1)  # (1, num_quantiles)
    loss = torch.where(
        errors >= 0,
        quantiles * errors,
        (quantiles - 1) * errors
    )
    return loss.mean()


def losses_to_string(
    losses: Union[np.ndarray, List[float]],
    precision: int = 3,
) -> str:
    """
    Convert an array of loss values to a compact string representation.
    
    Steps:
    1. Divide all values by the maximum (percentage of "loss to go")
    2. Round to fixed precision (e.g., 3 decimals)
    3. Convert to string without the "0." prefix
    
    Example:
        [3.14, 3.11, 3.09] -> "996, 972, 951"
        
    Args:
        losses: Array of loss values
        precision: Number of decimal places (default: 3)
        
    Returns:
        Comma-separated string of integer representations
    """
    losses = np.asarray(losses, dtype=np.float64)
    
    # Handle edge cases
    if len(losses) == 0:
        return ""
    
    max_val = np.max(losses)
    if max_val == 0:
        # All zeros case
        return ", ".join(["0" * precision] * len(losses))
    
    # Divide by max to get percentages (0 to 1)
    normalized = losses / max_val
    
    # Round to fixed precision and convert to integers
    # e.g., 0.996 with precision=3 -> 996
    multiplier = 10 ** precision
    as_ints = np.round(normalized * multiplier).astype(int)
    
    # Convert to zero-padded strings
    format_str = f"{{:0{precision}d}}"
    formatted = [format_str.format(v) for v in as_ints]
    
    return ", ".join(formatted)


def string_to_losses(
    s: str,
    max_val: float,
    precision: int = 3,
) -> np.ndarray:
    """
    Convert a string representation back to loss values.
    
    This is the inverse of losses_to_string, but requires the original
    max value to reconstruct the actual losses.
    
    Example:
        "996, 972, 951" with max_val=3.14 -> [3.14, 3.054, 2.986]
        
    Args:
        s: Comma-separated string of integer representations
        max_val: The maximum value used during normalization
        precision: Number of decimal places used (default: 3)
        
    Returns:
        Array of reconstructed loss values
    """
    if not s.strip():
        return np.array([])
    
    # Parse integers from string
    parts = [p.strip() for p in s.split(",")]
    as_ints = np.array([int(p) for p in parts], dtype=np.float64)
    
    # Convert back to normalized values
    multiplier = 10 ** precision
    normalized = as_ints / multiplier
    
    # Reconstruct original scale
    return normalized * max_val


def _extract_step_number(step_dir_name: str) -> Optional[int]:
    """Extract step number from directory name like 'step1250-seed-default'."""
    match = re.match(r"step(\d+)", step_dir_name)
    if match:
        return int(match.group(1))
    return None


def preprocess_model_losses(
    model_dir: str,
    output_filename: str = "avg_losses_by_step.npy",
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess loss data for a single model.
    
    For each step directory:
    1. Load all .npy files
    2. Compute nanmean across all values (ignoring NaN)
    3. Collect into a loss vector sorted by step
    4. Replace any remaining NaN with 0
    5. Save to model directory
    
    Args:
        model_dir: Path to model directory (e.g., results/datadecide_eval/DataDecide-c4-90M)
        output_filename: Name of output .npy file
        overwrite: Whether to overwrite existing output file
        
    Returns:
        Tuple of (steps array, losses array)
    """
    model_dir = Path(model_dir)
    output_path = model_dir / output_filename
    
    # Check if already processed
    if output_path.exists() and not overwrite:
        logging.info(f"Skipping {model_dir.name} - already processed")
        data = np.load(output_path)
        # Assume saved as 2D array: [[steps], [losses]]
        return data[0], data[1]
    
    step_losses = []
    
    # Find all step directories
    for item in model_dir.iterdir():
        if not item.is_dir():
            continue
            
        step_num = _extract_step_number(item.name)
        if step_num is None:
            continue
        
        # Find all .npy files in this step directory
        npy_files = list(item.glob("*.npy"))
        if not npy_files:
            continue
        
        # Load and concatenate all losses from this step
        all_losses = []
        for npy_file in npy_files:
            losses = np.load(npy_file)
            all_losses.append(losses.flatten())
        
        all_losses = np.concatenate(all_losses)
        
        # Compute mean ignoring NaN
        avg_loss = np.nanmean(all_losses)
        step_losses.append((step_num, avg_loss))
    
    if not step_losses:
        logging.warning(f"No step directories found in {model_dir}")
        return np.array([]), np.array([])
    
    # Sort by step number
    step_losses.sort(key=lambda x: x[0])
    
    steps = np.array([s[0] for s in step_losses], dtype=np.int64)
    losses = np.array([s[1] for s in step_losses], dtype=np.float64)
    
    # Replace NaN with 0 (after averaging)
    losses = np.nan_to_num(losses, nan=0.0)
    
    # Save as 2D array: [[steps], [losses]]
    output_data = np.stack([steps.astype(np.float64), losses])
    np.save(output_path, output_data)
    
    logging.info(f"Saved {len(steps)} step losses to {output_path}")
    
    return steps, losses


def preprocess_model_loss_histograms(
    model_dir: str,
    output_filename: str = "histograms_by_step.npz",
    num_bins: int = 32,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    inverse_perplexity: bool = False,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess loss histograms for a single model.

    For each step directory:
    1. Load all .npy files
    2. Optionally apply inverse perplexity transform: x -> e^(-x)
    3. Compute normalized histogram
    4. Save to model directory as .npz with 'steps' and 'histograms' arrays

    Args:
        model_dir: Path to model directory (e.g., results/datadecide_eval/DataDecide-c4-90M)
        output_filename: Name of output .npz file
        num_bins: Number of histogram bins
        bin_min: Minimum value for binning (0.0 for inverse_perplexity)
        bin_max: Maximum value for binning (1.0 for inverse_perplexity, 15.0 otherwise)
        inverse_perplexity: If True, transform losses x -> e^(-x) before binning
        overwrite: Whether to overwrite existing output file

    Returns:
        Tuple of (steps array shape (N,), histograms array shape (N, num_bins))
    """
    model_dir = Path(model_dir)

    # Append _invp to filename if using inverse perplexity transform
    if inverse_perplexity:
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}_invp{ext}"

    output_path = model_dir / output_filename

    # Check if already processed
    if output_path.exists() and not overwrite:
        logging.info(f"Skipping {model_dir.name} - already processed")
        data = np.load(output_path)
        return data['steps'], data['histograms']

    step_histograms = []

    # Find all step directories
    for item in model_dir.iterdir():
        if not item.is_dir():
            continue

        step_num = _extract_step_number(item.name)
        if step_num is None:
            continue

        # Find all .npy files in this step directory
        npy_files = list(item.glob("*.npy"))
        if not npy_files:
            continue

        # Load and concatenate all losses from this step
        all_losses = []
        for npy_file in npy_files:
            losses = np.load(npy_file)
            all_losses.append(losses.flatten())

        all_losses = np.concatenate(all_losses)

        # Clean losses
        all_losses = np.nan_to_num(all_losses, nan=0.0, posinf=10.0, neginf=0.0)

        # Apply inverse perplexity transform if enabled: x -> e^(-x)
        if inverse_perplexity:
            all_losses = np.exp(-all_losses)

        all_losses = np.clip(all_losses, bin_min, bin_max - 1e-6)

        # Compute normalized histogram
        hist, _ = np.histogram(all_losses, bins=num_bins, range=(bin_min, bin_max))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)  # Normalize to sum to 1

        step_histograms.append((step_num, hist))

    if not step_histograms:
        logging.warning(f"No step directories found in {model_dir}")
        return np.array([]), np.array([])

    # Sort by step number
    step_histograms.sort(key=lambda x: x[0])

    steps = np.array([s[0] for s in step_histograms], dtype=np.int64)
    histograms = np.stack([s[1] for s in step_histograms])  # (N, num_bins)

    # Save as .npz
    np.savez(output_path, steps=steps, histograms=histograms)

    logging.info(f"Saved {len(steps)} step histograms to {output_path}")

    return steps, histograms


def preprocess_all_model_histograms(
    base_dir: str = "results/datadecide_eval",
    output_filename: str = "histograms_by_step.npz",
    num_bins: int = 32,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    inverse_perplexity: bool = False,
    overwrite: bool = False,
) -> dict:
    """
    Preprocess loss histograms for all models in the base directory.

    Args:
        base_dir: Base directory containing model directories
        output_filename: Name of output .npz file for each model
        num_bins: Number of histogram bins
        bin_min: Minimum value for binning (0.0 for inverse_perplexity)
        bin_max: Maximum value for binning (1.0 for inverse_perplexity, 15.0 otherwise)
        inverse_perplexity: If True, transform losses x -> e^(-x) before binning
        overwrite: Whether to overwrite existing output files

    Returns:
        Dictionary mapping model name to (steps, histograms) tuple
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    # Compute actual output filename for logging (same logic as preprocess_model_loss_histograms)
    actual_output_filename = output_filename
    if inverse_perplexity:
        base, ext = os.path.splitext(output_filename)
        actual_output_filename = f"{base}_invp{ext}"

    results = {}
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    for model_dir in tqdm(model_dirs, desc=f"Processing model histograms -> {actual_output_filename}"):
        try:
            steps, histograms = preprocess_model_loss_histograms(
                model_dir,
                output_filename=output_filename,
                num_bins=num_bins,
                bin_min=bin_min,
                bin_max=bin_max,
                inverse_perplexity=inverse_perplexity,
                overwrite=overwrite,
            )
            results[model_dir.name] = (steps, histograms)
        except Exception as e:
            logging.error(f"Error processing {model_dir.name}: {e}")
            continue

    logging.info(f"Processed histograms for {len(results)} models")
    return results


def preprocess_all_models(
    base_dir: str = "results/datadecide_eval",
    output_filename: str = "avg_losses_by_step.npy",
    overwrite: bool = False,
) -> dict:
    """
    Preprocess loss data for all models in the base directory.
    
    Args:
        base_dir: Base directory containing model directories
        output_filename: Name of output .npy file for each model
        overwrite: Whether to overwrite existing output files
        
    Returns:
        Dictionary mapping model name to (steps, losses) tuple
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    results = {}
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for model_dir in tqdm(model_dirs, desc="Processing models"):
        try:
            steps, losses = preprocess_model_losses(
                model_dir,
                output_filename=output_filename,
                overwrite=overwrite,
            )
            results[model_dir.name] = (steps, losses)
        except Exception as e:
            logging.error(f"Error processing {model_dir.name}: {e}")
            continue
    
    logging.info(f"Processed {len(results)} models")
    return results


def load_preprocessed_losses(
    base_dir: str = "results/datadecide_eval",
    filename: str = "avg_losses_by_step.npy",
) -> dict:
    """
    Load preprocessed loss data for all models.
    
    Args:
        base_dir: Base directory containing model directories
        filename: Name of the preprocessed .npy file
        
    Returns:
        Dictionary mapping model name to (steps, losses) tuple
    """
    base_dir = Path(base_dir)
    results = {}
    
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        loss_file = model_dir / filename
        if not loss_file.exists():
            continue
        
        data = np.load(loss_file)
        steps = data[0].astype(np.int64)
        losses = data[1]
        results[model_dir.name] = (steps, losses)
    
    return results


def parse_model_dir_name(dir_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model directory name to extract data source and params.
    
    Expected format: DataDecide-{data_source}-{params}
    Examples:
        - DataDecide-c4-150M -> ('c4', '150M')
        - DataDecide-dclm-baseline-1B -> ('dclm-baseline', '1B')
        - DataDecide-c4-1B -> ('c4', '1B')
    
    Args:
        dir_name: Directory name (e.g., 'DataDecide-c4-150M')
        
    Returns:
        Tuple of (data_source, params) or (None, None) if parsing fails
    """
    # Pattern: DataDecide-{data_source}-{params}
    # params is typically like 90M, 150M, 300M, 530M, 750M, 1B
    match = re.match(r"DataDecide-(.+)-(\d+[MB])", dir_name)
    if match:
        data_source = match.group(1)
        params = match.group(2)
        return data_source, params
    return None, None


# Mapping from simplified directory names to HuggingFace dataset 'data' column values
# This can be extended as needed
DATA_SOURCE_MAPPING = {
    "c4": "C4",
    "dclm-baseline": "DCLM-Baseline",
    "dclm-baseline-qc-7p-fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm-baseline-qc-7p-fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm-baseline-qc-fw-3p": "DCLM-Baseline (QC FW 3%)",
    "dclm-baseline-qc-fw-10p": "DCLM-Baseline (QC FW 10%)",
    "dclm-baseline-qc-10p": "DCLM-Baseline (QC 10%)",
    "dclm-baseline-qc-20p": "DCLM-Baseline (QC 20%)",
    "dclm-baseline-75p-dolma1.7-25p": "DCLM-Baseline 75% / Dolma 25%",
    "dclm-baseline-50p-dolma1.7-50p": "DCLM-Baseline 50% / Dolma 50%",
    "dclm-baseline-25p-dolma1.7-75p": "DCLM-Baseline 25% / Dolma 75%",
    "dolma": "Dolma",
    "dolma1_6plus": "Dolma1.6++",
    "dolma1_7": "Dolma1.7",
    "dolma1_7-no-code": "Dolma1.7 (no code)",
    "dolma1_7-no-flan": "Dolma1.7 (no Flan)",
    "dolma1_7-no-math-code": "Dolma1.7 (no math, code)",
    "dolma1_7-no-reddit": "Dolma1.7 (no Reddit)",
    "falcon": "Falcon",
    "falcon-and-cc": "Falcon+CC",
    "falcon-and-cc-qc-10p": "Falcon+CC (QC 10%)",
    "falcon-and-cc-qc-20p": "Falcon+CC (QC 20%)",
    "falcon-and-cc-qc-orig-10p": "Falcon+CC (QC Orig 10%)",
    "falcon-and-cc-qc-tulu-10p": "Falcon+CC (QC Tulu 10%)",
    "fineweb": "FineWeb",
    "fineweb-edu": "FineWeb-Edu",
    "fineweb-pro": "FineWeb-Pro",
    "redpajama": "RedPajama",
}


def match_data_source(simplified_name: str, available_data_sources: List[str]) -> Optional[str]:
    """
    Match a simplified data source name to the full dataset name.
    
    Uses the DATA_SOURCE_MAPPING first, then falls back to case-insensitive prefix matching.
    
    Args:
        simplified_name: Simplified name from directory (e.g., 'c4', 'dclm-baseline')
        available_data_sources: List of available data source names from the dataset
        
    Returns:
        Matched full data source name, or None if no match found
    """
    # Normalize the simplified name
    normalized = simplified_name.lower().replace("_", "-")
    
    # Check direct mapping first (normalize keys too for matching)
    for key, mapped in DATA_SOURCE_MAPPING.items():
        key_normalized = key.lower().replace("_", "-")
        if normalized == key_normalized:
            # Find exact or prefix match in available sources
            for source in available_data_sources:
                if source == mapped or source.startswith(mapped):
                    return source
            break
    
    # Fall back to case-insensitive prefix matching
    for source in available_data_sources:
        source_normalized = source.lower().replace("_", "-").replace(" ", "-")
        if source_normalized.startswith(normalized):
            return source
        # Also check if the source starts with the normalized name (ignoring parenthetical suffixes)
        source_base = source.split("(")[0].strip().lower().replace("_", "-").replace(" ", "-")
        if source_base == normalized or source_base.startswith(normalized):
            return source
    
    return None


def extract_accuracies_for_model(
    model_dir: str,
    dataset_df: "pd.DataFrame" = None,
    seed: str = "default",
    overwrite: bool = False,
) -> dict:
    """
    Extract accuracy values for a single model from the DataDecide evaluation dataset.
    
    For each task, saves a numpy array with shape (n_steps,) containing acc_raw values
    sorted by step number.
    
    Args:
        model_dir: Path to model directory (e.g., results/datadecide_eval/DataDecide-c4-90M)
        dataset_df: Pre-loaded pandas DataFrame from the HuggingFace dataset.
                   If None, will load the dataset.
        seed: Seed value to filter by (default: 'default')
        overwrite: Whether to overwrite existing output files
        
    Returns:
        Dictionary mapping task name to (steps, accuracies) tuple
    """
    import pandas as pd
    
    model_dir = Path(model_dir)
    dir_name = model_dir.name
    
    # Parse directory name
    data_source, params = parse_model_dir_name(dir_name)
    if data_source is None or params is None:
        logging.warning(f"Could not parse directory name: {dir_name}")
        return {}
    
    # Load dataset if not provided
    if dataset_df is None:
        try:
            from datasets import load_dataset
            dataset_df = load_dataset("allenai/DataDecide-eval-results", split="train").to_pandas()
        except Exception as e:
            logging.error(f"Failed to load DataDecide dataset: {e}")
            return {}
    
    # Get available data sources and match
    available_data_sources = dataset_df["data"].unique().tolist()
    matched_data = match_data_source(data_source, available_data_sources)
    
    if matched_data is None:
        logging.warning(f"Could not match data source '{data_source}' to dataset. "
                       f"Available: {available_data_sources}")
        return {}
    
    logging.info(f"Matched '{data_source}' -> '{matched_data}'")
    
    # Filter dataset
    filtered = dataset_df[
        (dataset_df["params"] == params) &
        (dataset_df["data"] == matched_data) &
        (dataset_df["seed"] == seed)
    ].copy()
    
    if len(filtered) == 0:
        logging.warning(f"No data found for params={params}, data={matched_data}, seed={seed}")
        return {}
    
    # Extract accuracy from metrics column
    filtered["accuracy"] = filtered["metrics"].apply(
        lambda x: eval(x).get("acc_raw", np.nan) if isinstance(x, str) else np.nan
    )
    
    # Get unique tasks
    tasks = filtered["task"].unique()
    results = {}
    
    for task in tasks:
        task_data = filtered[filtered["task"] == task].sort_values("step")
        
        # Check if already exists
        output_path = model_dir / f"{task}.npy"
        if output_path.exists() and not overwrite:
            logging.info(f"Skipping {task} - already exists")
            data = np.load(output_path)
            results[task] = (data[0].astype(np.int64), data[1])
            continue
        
        steps = task_data["step"].values.astype(np.int64)
        accuracies = task_data["accuracy"].values.astype(np.float64)
        
        # Replace NaN with 0
        accuracies = np.nan_to_num(accuracies, nan=0.0)
        
        # Save as 2D array: [[steps], [accuracies]]
        output_data = np.stack([steps.astype(np.float64), accuracies])
        np.save(output_path, output_data)
        
        logging.info(f"Saved {task}.npy with {len(steps)} steps")
        results[task] = (steps, accuracies)
    
    return results


def extract_accuracies_all_models(
    base_dir: str = "results/datadecide_eval",
    seed: str = "default",
    overwrite: bool = False,
) -> dict:
    """
    Extract accuracy values for all models in the base directory.
    
    Loads the DataDecide evaluation dataset once and processes all model directories.
    
    Args:
        base_dir: Base directory containing model directories
        seed: Seed value to filter by (default: 'default')
        overwrite: Whether to overwrite existing output files
        
    Returns:
        Dictionary mapping model name to dict of task -> (steps, accuracies)
    """
    import pandas as pd
    from datasets import load_dataset
    
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Load dataset once
    logging.info("Loading DataDecide evaluation dataset...")
    dataset_df = load_dataset("allenai/DataDecide-eval-results", split="train").to_pandas()
    logging.info(f"Loaded {len(dataset_df)} rows")
    
    results = {}
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for model_dir in tqdm(model_dirs, desc="Extracting accuracies"):
        try:
            task_results = extract_accuracies_for_model(
                model_dir,
                dataset_df=dataset_df,
                seed=seed,
                overwrite=overwrite,
            )
            results[model_dir.name] = task_results
        except Exception as e:
            logging.error(f"Error processing {model_dir.name}: {e}")
            continue
    
    logging.info(f"Processed {len(results)} models")
    return results


def load_preprocessed_accuracies(
    model_dir: str,
    tasks: Optional[List[str]] = None,
) -> dict:
    """
    Load preprocessed accuracy data for a single model.
    
    Args:
        model_dir: Path to model directory
        tasks: Optional list of task names to load. If None, loads all available.
        
    Returns:
        Dictionary mapping task name to (steps, accuracies) tuple
    """
    model_dir = Path(model_dir)
    results = {}
    
    # Find all task .npy files (exclude avg_losses_by_step.npy)
    for npy_file in model_dir.glob("*.npy"):
        if npy_file.name == "avg_losses_by_step.npy":
            continue
        
        task_name = npy_file.stem
        if tasks is not None and task_name not in tasks:
            continue
        
        data = np.load(npy_file)
        steps = data[0].astype(np.int64)
        accuracies = data[1]
        results[task_name] = (steps, accuracies)
    
    return results


def main():
    """Run preprocessing on all models."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess loss data for meta-loss predictor")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="results/datadecide_eval",
        help="Base directory containing model directories",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="avg_losses_by_step.npy",
        help="Name of output .npy file for each model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--extract_accuracies",
        action="store_true",
        help="Extract accuracies from DataDecide eval dataset instead of losses",
    )
    parser.add_argument(
        "--preprocess_histograms",
        action="store_true",
        help="Preprocess loss histograms instead of average losses",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=32,
        help="Number of histogram bins (default: 32)",
    )
    parser.add_argument(
        "--bin_min",
        type=float,
        default=0.0,
        help="Minimum loss value for histogram binning (default: 0.0)",
    )
    parser.add_argument(
        "--bin_max",
        type=float,
        default=15.0,
        help="Maximum loss value for histogram binning (default: 15.0, use 1.0 with --inverse_perplexity)",
    )
    parser.add_argument(
        "--inverse_perplexity",
        action="store_true",
        help="Transform losses x -> e^(-x) before binning (values become 0-1 range)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="default",
        help="Seed value to filter by when extracting accuracies",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.extract_accuracies:
        results = extract_accuracies_all_models(
            base_dir=args.data_dir,
            seed=args.seed,
            overwrite=args.overwrite,
        )

        # Print summary
        print(f"\nExtracted accuracies for {len(results)} models:")
        for model_name, task_results in sorted(results.items()):
            if task_results:
                tasks = list(task_results.keys())
                print(f"  {model_name}: {len(tasks)} tasks")
    elif args.preprocess_histograms:
        results = preprocess_all_model_histograms(
            base_dir=args.data_dir,
            output_filename="histograms_by_step.npz",
            num_bins=args.num_bins,
            bin_min=args.bin_min,
            bin_max=args.bin_max,
            inverse_perplexity=args.inverse_perplexity,
            overwrite=args.overwrite,
        )

        # Print summary
        print(f"\nProcessed histograms for {len(results)} models:")
        for model_name, (steps, histograms) in sorted(results.items()):
            if len(steps) > 0:
                print(f"  {model_name}: {len(steps)} steps, histogram shape {histograms.shape}")
    else:
        results = preprocess_all_models(
            base_dir=args.data_dir,
            output_filename=args.output_filename,
            overwrite=args.overwrite,
        )

        # Print summary
        print(f"\nProcessed {len(results)} models:")
        for model_name, (steps, losses) in sorted(results.items()):
            if len(steps) > 0:
                print(f"  {model_name}: {len(steps)} steps, loss range [{losses.min():.4f}, {losses.max():.4f}]")


if __name__ == "__main__":
    main()
