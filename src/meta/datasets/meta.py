"""
Meta-prediction datasets for training neural scaling law predictors.

This module provides:
- MetaLossDataset: Predicts loss values from temporal context
- MetaAccuracyDataset: Predicts task accuracies from temporal context
- MetaTaskAccuracyDataset: Diagnostic dataset with task accuracies as encoder input
- MetaLossEvalDataset: Evaluation dataset for comprehensive loss prediction
- MetaAccuracyDatasetAvgLoss: Uses average losses as encoder input
- collate_fn: Collate function for variable-length inputs
- collate_fn_accuracy: Collate function with task names
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch

from src.meta.datasets.base import MetaDataset


# =============================================================================
# Shared Utility Functions
# =============================================================================


def get_task_files(
    model_dir: Path,
    target_list: Optional[List[str]] = None,
    heldout_list: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Get task .npy files for a model directory.

    Args:
        model_dir: Path to model directory
        target_list: If specified, only return these tasks
        heldout_list: List of task names to exclude (e.g., ["avg_losses_by_step"])

    Returns:
        Dict mapping task name to file path
    """
    heldout = heldout_list or []
    task_files = {}
    for npy_file in model_dir.glob("*.npy"):
        task_name = npy_file.stem
        if target_list is not None:
            if task_name not in target_list:
                continue
        else:
            if task_name in heldout:
                continue
        task_files[task_name] = npy_file
    return task_files


def get_accuracy_context_and_target(
    values: np.ndarray,
    context_end_idx: int,
    target_idx: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Get context accuracies and target accuracy for accuracy prediction datasets.

    Returns:
        Tuple of (context_values, target_value, norm_factor=1.0)
    """
    context_values = values[:context_end_idx + 1].copy()
    target_value = values[target_idx]

    context_values = np.round(context_values, decimals=3)
    target_value = round(float(target_value), 3)

    return context_values.astype(np.float32), target_value, 1.0


# =============================================================================
# Dataset Classes
# =============================================================================


class MetaLossDataset(MetaDataset):
    """
    Dataset for meta-loss prediction (regression).

    Each sample provides:
    - context_values: Observed loss values (normalized by max)
    - context_gaps: Gap before each observed loss
    - query_gap: How far ahead to predict from the last context observation
    - target_value: The loss value to predict (normalized)
    - encoder_input: Raw token-level losses for conditioning the soft prompt encoder
    """

    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all valid (model, context_end_idx, target_idx) triplets.

        For each model, we create samples where:
        - context_end_idx: Last observed context step
        - target_idx: Index of the target loss to predict
        - query_gap: target_idx - context_end_idx
        """
        from src.meta.utils import _extract_step_number

        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            avg_loss_file = model_dir / "avg_losses_by_step.npy"
            if not avg_loss_file.exists():
                logging.warning(f"No preprocessed losses for {model_dir.name}")
                continue

            data = np.load(avg_loss_file)
            steps = data[0].astype(np.int64)
            avg_losses = data[1]

            # Filter out step 0 if requested
            if self.exclude_step0:
                mask = steps != 0
                steps = steps[mask]
                avg_losses = avg_losses[mask]

            if len(steps) < self.min_context_steps + 1:
                continue

            # Find raw loss files for each step
            step_to_npy = {}
            for item in model_dir.iterdir():
                if item.is_dir():
                    extracted = _extract_step_number(item.name)
                    if extracted is not None:
                        npy_files = list(item.glob(self.loss_file_pattern))
                        if npy_files:
                            step_to_npy[extracted] = npy_files

            # Create samples: for each valid (context_end, target) pair
            for context_end_idx in range(self.min_context_steps - 1, len(steps) - 1):
                # Get the step number at context_end
                context_step = steps[context_end_idx]

                # Check if we have raw losses for this step
                if context_step not in step_to_npy:
                    continue

                # Create samples for each valid target
                max_target_idx = min(context_end_idx + self.max_query_gap + 1, len(steps))
                for target_idx in range(context_end_idx + 1, max_target_idx):
                    query_gap = target_idx - context_end_idx

                    samples.append({
                        "model_dir": model_dir,
                        "model_name": model_dir.name,
                        "context_end_idx": context_end_idx,
                        "target_idx": target_idx,
                        "query_gap": query_gap,
                        "steps": steps,
                        "values": avg_losses,
                        "raw_loss_files": step_to_npy[context_step],
                    })

        logging.info(f"Found {len(samples)} valid loss samples across all models")
        return samples

    def _get_context_and_target(self, sample: Dict) -> Tuple[np.ndarray, float, float]:
        """Get normalized context losses and target loss."""
        context_end_idx = sample["context_end_idx"]
        target_idx = sample["target_idx"]
        values = sample["values"]

        # Context values (up to and including context_end_idx)
        context_values = values[:context_end_idx + 1].copy()
        target_value = values[target_idx]

        # Normalize by max of the full sequence
        max_val = np.max(values)
        context_normalized = context_values / max_val
        target_normalized = target_value / max_val

        return context_normalized.astype(np.float32), float(target_normalized), float(max_val)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Override to use backward-compatible field names."""
        sample = self.samples[idx]

        target_idx = sample["target_idx"]

        # Get context and target
        context_values, target_value, norm_factor = self._get_context_and_target(sample)
        context_indices = np.arange(len(context_values))

        # Apply dropout for data augmentation (only during training)
        if self.drop_prob > 0 and self.split == "train":
            context_values, context_indices = self._apply_dropout(context_values)

        # Compute forward gaps (each gap[i] = distance from value[i] to value[i+1] or target)
        context_gaps = self._compute_gaps(context_indices, target_idx)

        # Load encoder input
        encoder_input = self._load_encoder_input(sample)

        # Use backward-compatible field names with torch.from_numpy for efficiency
        return {
            "encoder_input": torch.from_numpy(encoder_input.copy()),  # copy needed since cache is shared
            "context_losses": torch.from_numpy(context_values),
            "context_gaps": torch.from_numpy(context_gaps),
            "target_loss": torch.tensor(target_value, dtype=torch.float32),
            "max_loss": torch.tensor(norm_factor, dtype=torch.float32),
            "model_name": sample["model_name"],
        }


class MetaAccuracyDataset(MetaDataset):
    """
    Dataset for meta-accuracy prediction (regression).

    Each sample provides:
    - context_values: Observed accuracy values (already 0-1, no normalization needed)
    - context_gaps: Gap before each observed accuracy
    - query_gap: How far ahead to predict from the last context observation
    - target_value: The accuracy value to predict
    - encoder_input: Raw token-level losses for conditioning (same as MetaLossDataset)

    Note: This dataset is larger than MetaLossDataset because each encoder_input
    (raw losses at a given step) is paired with targets from all tasks (or target_list if specified).
    """

    def __init__(
        self,
        data_dir: str,
        min_context_steps: int = 4,
        max_query_gap: int = 10,
        max_encoder_tokens: int = 8192,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        drop_prob: float = 0.0,
        target_list: Optional[List[str]] = None,
        heldout_list: Optional[List[str]] = None,
        inverse_perplexity: bool = False,
        bin_min: float = 0.0,
        bin_max: float = 15.0,
        _shared_samples: Optional[List[Dict]] = None,
    ):
        """
        Args:
            data_dir: Base directory containing model directories
            min_context_steps: Minimum number of context observations
            max_query_gap: Maximum gap ahead to predict
            max_encoder_tokens: Maximum number of raw loss values for soft prompt encoder
            split: "train" or "val"
            train_ratio: Ratio of data for training
            seed: Random seed for splitting
            drop_prob: Probability of dropping intermediate context accuracies (data augmentation)
            target_list: If specified, only predict these tasks. If None, samples from all tasks.
            heldout_list: List of task names to exclude from loading. Defaults to ["avg_losses_by_step"].
            inverse_perplexity: If True, transform losses x -> e^(-x) (values become 0-1)
            bin_min: Minimum value for clipping (0.0 for inverse_perplexity)
            bin_max: Maximum value for clipping (1.0 for inverse_perplexity, 15.0 otherwise)
            _shared_samples: Internal parameter to share pre-built samples between train/val
        """
        self.target_list = target_list
        self.heldout_list = heldout_list if heldout_list is not None else ["avg_losses_by_step"]
        self._task_files_cache = {}  # Cache for task files per model
        super().__init__(
            data_dir=data_dir,
            min_context_steps=min_context_steps,
            max_query_gap=max_query_gap,
            max_encoder_tokens=max_encoder_tokens,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            drop_prob=drop_prob,
            inverse_perplexity=inverse_perplexity,
            bin_min=bin_min,
            bin_max=bin_max,
            _shared_samples=_shared_samples,
        )

    def _get_task_files(self, model_dir: Path) -> Dict[str, Path]:
        """Get task .npy files for a model directory (cached)."""
        if model_dir not in self._task_files_cache:
            self._task_files_cache[model_dir] = get_task_files(
                model_dir, self.target_list, self.heldout_list
            )
        return self._task_files_cache[model_dir]

    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all valid (model, task, context_end_idx, target_idx) tuples.

        For each model and task, we create samples where:
        - context_end_idx: Last observed context step
        - target_idx: Index of the target accuracy to predict
        - query_gap: target_idx - context_end_idx
        """
        from src.meta.utils import _extract_step_number

        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Get all task files
            task_files = self._get_task_files(model_dir)
            if not task_files:
                logging.warning(f"No task accuracy files for {model_dir.name}")
                continue

            # Find raw loss files for each step (for encoder input)
            step_to_npy = {}
            for item in model_dir.iterdir():
                if item.is_dir():
                    extracted = _extract_step_number(item.name)
                    if extracted is not None:
                        npy_files = list(item.glob(self.loss_file_pattern))
                        if npy_files:
                            step_to_npy[extracted] = npy_files

            # Load all task data to find common steps
            tasks_to_process = list(task_files.keys())
            task_data = {}
            common_steps = None

            for task_name in tasks_to_process:
                data = np.load(task_files[task_name])
                steps = data[0].astype(np.int64)
                accuracies = data[1]
                task_data[task_name] = {"steps": steps, "accuracies": accuracies}

                if common_steps is None:
                    common_steps = set(steps)
                else:
                    common_steps &= set(steps)

            if common_steps is None or len(common_steps) < self.min_context_steps + 1:
                continue

            # Sort common steps
            common_steps = sorted(common_steps)

            # Filter out step 0 if requested
            if self.exclude_step0:
                common_steps = [s for s in common_steps if s != 0]

            if len(common_steps) < self.min_context_steps + 1:
                continue

            # Create step-to-index mapping for each task
            for task_name, data in task_data.items():
                step_to_idx = {s: i for i, s in enumerate(data["steps"])}
                # Remap to common steps
                common_accuracies = np.array([
                    data["accuracies"][step_to_idx[s]] for s in common_steps
                ])
                # Round to 3 decimal points
                common_accuracies = np.round(common_accuracies, decimals=3)
                task_data[task_name]["common_accuracies"] = common_accuracies

            # Create samples for each task
            for task_name in tasks_to_process:
                accuracies = task_data[task_name]["common_accuracies"]

                for context_end_idx in range(self.min_context_steps - 1, len(common_steps) - 1):
                    # Get the step number at context_end
                    context_step = common_steps[context_end_idx]

                    # Check if we have raw losses for this step
                    if context_step not in step_to_npy:
                        continue

                    max_target_idx = min(context_end_idx + self.max_query_gap + 1, len(common_steps))

                    for target_idx in range(context_end_idx + 1, max_target_idx):
                        query_gap = target_idx - context_end_idx

                        samples.append({
                            "model_dir": model_dir,
                            "model_name": model_dir.name,
                            "task_name": task_name,
                            "context_end_idx": context_end_idx,
                            "target_idx": target_idx,
                            "query_gap": query_gap,
                            "steps": np.array(common_steps),
                            "values": accuracies,
                            "raw_loss_files": step_to_npy[context_step],
                        })

        logging.info(f"Found {len(samples)} valid accuracy samples across all models")
        return samples

    def _get_context_and_target(self, sample: Dict) -> Tuple[np.ndarray, float, float]:
        """Get context accuracies and target accuracy (no normalization needed)."""
        return get_accuracy_context_and_target(
            sample["values"], sample["context_end_idx"], sample["target_idx"]
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Override to add task_name to output."""
        result = super().__getitem__(idx)
        result["task_name"] = self.samples[idx]["task_name"]
        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length inputs.

    Works for both MetaLossDataset (with backward-compatible field names)
    and MetaAccuracyDataset (with new field names).
    """
    # Detect which field names are used
    use_old_names = "context_losses" in batch[0]
    context_key = "context_losses" if use_old_names else "context_values"
    target_key = "target_loss" if use_old_names else "target_value"
    norm_key = "max_loss" if use_old_names else "norm_factor"

    # Handle encoder inputs - may be variable-length for hist_full and avg_loss
    encoder_inputs = [item["encoder_input"] for item in batch]

    # Check if encoder inputs need padding (variable-length case)
    encoder_shapes = [p.shape for p in encoder_inputs]
    if len(encoder_shapes[0]) == 1:
        # 1D case (avg_loss): (num_losses,) - pad to max length
        max_encoder_len = max(p.shape[0] for p in encoder_inputs)
        padded_encoder = []
        encoder_mask = []
        for p in encoder_inputs:
            pad_len = max_encoder_len - p.shape[0]
            if pad_len > 0:
                p_padded = torch.nn.functional.pad(p, (0, pad_len), value=0.0)
                mask = torch.cat([torch.ones(p.shape[0]), torch.zeros(pad_len)])
            else:
                p_padded = p
                mask = torch.ones(p.shape[0])
            padded_encoder.append(p_padded)
            encoder_mask.append(mask)
        stacked_encoder = torch.stack(padded_encoder)
        stacked_encoder_mask = torch.stack(encoder_mask)
    elif len(encoder_shapes[0]) == 2:
        if all(s == encoder_shapes[0] for s in encoder_shapes):
            stacked_encoder = torch.stack(encoder_inputs)
            stacked_encoder_mask = None
        else:
            # Variable-length (hist_full): (num_steps, num_bins) - pad first dim
            max_steps = max(p.shape[0] for p in encoder_inputs)
            num_bins = encoder_shapes[0][1]
            padded_encoder = []
            encoder_mask = []
            for p in encoder_inputs:
                pad_len = max_steps - p.shape[0]
                if pad_len > 0:
                    p_padded = torch.nn.functional.pad(p, (0, 0, 0, pad_len), value=0.0)
                    mask = torch.cat([torch.ones(p.shape[0]), torch.zeros(pad_len)])
                else:
                    p_padded = p
                    mask = torch.ones(p.shape[0])
                padded_encoder.append(p_padded)
                encoder_mask.append(mask)
            stacked_encoder = torch.stack(padded_encoder)
            stacked_encoder_mask = torch.stack(encoder_mask)
    else:
        # Unknown shape - just try to stack
        stacked_encoder = torch.stack(encoder_inputs)
        stacked_encoder_mask = None

    # Pad context values and gaps
    max_context_len = max(item[context_key].shape[0] for item in batch)
    padded_context = []
    padded_gaps = []
    context_mask = []
    for item in batch:
        c = item[context_key]
        g = item["context_gaps"]
        pad_len = max_context_len - c.shape[0]
        if pad_len > 0:
            c = torch.nn.functional.pad(c, (0, pad_len), value=0.0)
            g = torch.nn.functional.pad(g, (0, pad_len), value=0)
            mask = torch.cat([torch.ones(item[context_key].shape[0]), torch.zeros(pad_len)])
        else:
            mask = torch.ones(c.shape[0])
        padded_context.append(c)
        padded_gaps.append(g)
        context_mask.append(mask)

    result = {
        "encoder_input": stacked_encoder,
        context_key: torch.stack(padded_context),
        "context_gaps": torch.stack(padded_gaps),
        "context_mask": torch.stack(context_mask),
        target_key: torch.stack([item[target_key] for item in batch]),
        norm_key: torch.stack([item[norm_key] for item in batch]),
    }

    # Add encoder mask if variable-length encoder inputs were padded
    if stacked_encoder_mask is not None:
        result["encoder_mask"] = stacked_encoder_mask

    return result


def collate_fn_accuracy(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function specifically for MetaAccuracyDataset."""
    result = collate_fn(batch)
    # Add task names if present
    if "task_name" in batch[0]:
        result["task_name"] = [item["task_name"] for item in batch]
    return result


class MetaTaskAccuracyDataset(MetaDataset):
    """
    Strongest diagnostic: task-level accuracies as soft prompt encoder input.

    The encoder input is simply the vector of all task accuracies at the current
    step. The model just needs to learn to pick out which accuracy value corresponds
    to the target task. This is trivially solvable if the soft prompt encoder works correctly.

    Encoder input: Vector of task accuracies [acc_task1, acc_task2, ..., acc_taskN]
    Context/Target: Task-level accuracies (same as MetaAccuracyDataset)

    If the model fails at this task, there is a fundamental issue with the
    soft prompt encoder/model architecture.
    """

    def __init__(
        self,
        data_dir: str,
        min_context_steps: int = 4,
        max_query_gap: int = 10,
        max_encoder_tokens: int = 8192,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        drop_prob: float = 0.0,
        target_list: Optional[List[str]] = None,
        heldout_list: Optional[List[str]] = None,
        _shared_samples: Optional[List[Dict]] = None,
    ):
        """
        Args:
            data_dir: Base directory containing model directories
            min_context_steps: Minimum number of context observations
            max_query_gap: Maximum gap ahead to predict
            max_encoder_tokens: Not used here (input is always num_tasks values)
            split: "train" or "val"
            train_ratio: Ratio of data for training
            seed: Random seed for splitting
            drop_prob: Probability of dropping intermediate context values
            target_list: If specified, only predict these tasks
            heldout_list: Tasks to exclude (defaults to ["avg_losses_by_step"])
            _shared_samples: Internal parameter to share pre-built samples between train/val
        """
        self.target_list = target_list
        self.heldout_list = heldout_list if heldout_list is not None else ["avg_losses_by_step"]
        self._task_files_cache = {}
        # Will be set during _build_sample_index
        self.task_names_sorted: Optional[List[str]] = None

        super().__init__(
            data_dir=data_dir,
            min_context_steps=min_context_steps,
            max_query_gap=max_query_gap,
            max_encoder_tokens=max_encoder_tokens,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            drop_prob=drop_prob,
            _shared_samples=_shared_samples,
        )

    def _get_task_files(self, model_dir: Path) -> Dict[str, Path]:
        """Get task .npy files for a model directory (cached)."""
        if model_dir not in self._task_files_cache:
            self._task_files_cache[model_dir] = get_task_files(
                model_dir, self.target_list, self.heldout_list
            )
        return self._task_files_cache[model_dir]

    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all valid (model, task, context_end_idx, target_idx) tuples.

        Similar to MetaAccuracyDataset but stores all task accuracies per step
        for use as soft prompt encoder input.
        """
        samples = []
        all_task_names = set()

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            task_files = self._get_task_files(model_dir)
            if not task_files:
                logging.debug(f"No task accuracy files for {model_dir.name}")
                continue

            all_task_names.update(task_files.keys())

            # Load all task data to find common steps
            tasks_to_process = sorted(task_files.keys())  # Sort for consistency
            task_data = {}
            common_steps = None

            for task_name in tasks_to_process:
                data = np.load(task_files[task_name])
                steps = data[0].astype(np.int64)
                accuracies = data[1]
                task_data[task_name] = {"steps": steps, "accuracies": accuracies}

                if common_steps is None:
                    common_steps = set(steps)
                else:
                    common_steps &= set(steps)

            if common_steps is None or len(common_steps) < self.min_context_steps + 1:
                continue

            common_steps = sorted(common_steps)

            # Filter out step 0 if requested
            if self.exclude_step0:
                common_steps = [s for s in common_steps if s != 0]

            if len(common_steps) < self.min_context_steps + 1:
                continue

            # Create step-to-index mapping and common_accuracies for each task
            for task_name, data in task_data.items():
                step_to_idx = {s: i for i, s in enumerate(data["steps"])}
                common_accuracies = np.array([
                    data["accuracies"][step_to_idx[s]] for s in common_steps
                ])
                common_accuracies = np.round(common_accuracies, decimals=3)
                task_data[task_name]["common_accuracies"] = common_accuracies

            # Build per-step accuracy matrix: (num_steps, num_tasks)
            # Shape: each row is the accuracy vector at that step
            step_accuracy_matrix = np.stack([
                task_data[t]["common_accuracies"] for t in tasks_to_process
            ], axis=1)  # (num_steps, num_tasks)

            # Create samples for each task
            for task_name in tasks_to_process:
                task_idx = tasks_to_process.index(task_name)
                accuracies = task_data[task_name]["common_accuracies"]

                for context_end_idx in range(self.min_context_steps - 1, len(common_steps) - 1):
                    max_target_idx = min(context_end_idx + self.max_query_gap + 1, len(common_steps))

                    for target_idx in range(context_end_idx + 1, max_target_idx):
                        query_gap = target_idx - context_end_idx

                        samples.append({
                            "model_dir": model_dir,
                            "model_name": model_dir.name,
                            "task_name": task_name,
                            "task_idx": task_idx,
                            "context_end_idx": context_end_idx,
                            "target_idx": target_idx,
                            "query_gap": query_gap,
                            "steps": np.array(common_steps),
                            "values": accuracies,
                            # Use TARGET step accuracies so encoder input contains the answer
                            "step_accuracies": step_accuracy_matrix[target_idx],
                            "task_names": tasks_to_process,
                        })

        # Store sorted task names for reference
        self.task_names_sorted = sorted(all_task_names)
        logging.info(f"Found {len(samples)} valid task-accuracy samples across all models")
        logging.info(f"Tasks used for encoder input: {self.task_names_sorted}")
        return samples

    def _load_encoder_input(self, sample: Dict) -> np.ndarray:
        """Load task-level accuracy vector as soft prompt encoder input."""
        accuracies = sample["step_accuracies"].copy()
        return accuracies.astype(np.float32)

    def _get_context_and_target(self, sample: Dict) -> Tuple[np.ndarray, float, float]:
        """Get context accuracies and target accuracy."""
        return get_accuracy_context_and_target(
            sample["values"], sample["context_end_idx"], sample["target_idx"]
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        result = super().__getitem__(idx)
        result["task_name"] = self.samples[idx]["task_name"]
        return result


class MetaLossEvalDataset(MetaLossDataset):
    """
    Dataset for comprehensive evaluation of meta-loss prediction.

    Unlike MetaLossDataset which randomly samples (context_end, target, gap) triplets,
    this dataset generates ALL possible predictions within each sequence.

    For a loss sequence [L0, L1, L2, L3], creates samples:
    - Context [L0] → predict L1 (if min_context_steps <= 1)
    - Context [L0, L1] → predict L2 (if min_context_steps <= 2)
    - Context [L0, L1, L2] → predict L3 (if min_context_steps <= 3)

    Each sample uses the encoder input from the last context step.
    This is useful for evaluation to see how prediction quality varies with context length.
    """

    def __init__(
        self,
        data_dir: str,
        min_context_steps: int = 1,
        max_encoder_tokens: int = 8192,
        split: str = "val",
        train_ratio: float = 0.95,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Base directory containing model directories
            min_context_steps: Minimum number of context observations
            max_encoder_tokens: Maximum number of raw loss values for soft prompt encoder
            split: "train" or "val"
            train_ratio: Ratio of data for training
            seed: Random seed for splitting
        """
        # Override max_query_gap to always be 1 for this dataset
        super().__init__(
            data_dir=data_dir,
            min_context_steps=min_context_steps,
            max_query_gap=1,  # Always predict next step
            max_encoder_tokens=max_encoder_tokens,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            drop_prob=0.0,  # No dropout for eval
        )

    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all valid context-target pairs within each sequence.

        For each model, creates a sample for every possible (context, next_target) pair.
        """
        from src.meta.utils import _extract_step_number

        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            avg_loss_file = model_dir / "avg_losses_by_step.npy"
            if not avg_loss_file.exists():
                logging.warning(f"No preprocessed losses for {model_dir.name}")
                continue

            data = np.load(avg_loss_file)
            steps = data[0].astype(np.int64)
            avg_losses = data[1]

            # Filter out step 0 if requested
            if self.exclude_step0:
                mask = steps != 0
                steps = steps[mask]
                avg_losses = avg_losses[mask]

            if len(steps) < self.min_context_steps + 1:
                continue

            # Find raw loss files for each step
            step_to_npy = {}
            for item in model_dir.iterdir():
                if item.is_dir():
                    extracted = _extract_step_number(item.name)
                    if extracted is not None:
                        npy_files = list(item.glob(self.loss_file_pattern))
                        if npy_files:
                            step_to_npy[extracted] = npy_files

            # For each possible context ending position, create a sample to predict the next step
            # This creates multiple overlapping samples from the same sequence
            for context_end_idx in range(self.min_context_steps - 1, len(steps) - 1):
                target_idx = context_end_idx + 1

                # Get the step number at context_end (this is where we get encoder input from)
                context_step = steps[context_end_idx]

                # Check if we have raw losses for this step
                if context_step not in step_to_npy:
                    continue

                query_gap = 1  # Always predict next step

                samples.append({
                    "model_dir": model_dir,
                    "model_name": model_dir.name,
                    "context_end_idx": context_end_idx,
                    "target_idx": target_idx,
                    "query_gap": query_gap,
                    "steps": steps,
                    "values": avg_losses,
                    "raw_loss_files": step_to_npy[context_step],
                })

        logging.info(f"Found {len(samples)} eval samples (all context-target pairs in each sequence)")
        return samples
    
class MetaAccuracyDatasetDeltaHist(MetaDataset):
    """
    Dataset for accuracy prediction using histogram deltas as soft prompt input.

    For each prediction, the encoder input is the delta histogram:
        delta_histogram = histogram[query_step] - histogram[context_end_step]

    This represents the change in the loss distribution from the last observed
    checkpoint to the target checkpoint we're trying to predict.

    The soft prompt generated from this delta is placed as a suffix after the
    context (temporal accuracy history), providing query-specific information.
    """

    def __init__(
        self,
        data_dir: str,
        min_context_steps: int = 4,
        max_query_gap: int = 10,
        max_encoder_tokens: int = 8192,  # Not used but kept for API compatibility
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        drop_prob: float = 0.0,
        target_list: Optional[List[str]] = None,
        heldout_list: Optional[List[str]] = None,
        loss_file_pattern: str = "word_losses*.npy",  # Not used
        max_clip: float = 20.0,  # Not used
        num_bins: int = 64,
        inverse_perplexity: bool = False,
        _shared_samples: Optional[List[Dict]] = None,
        _shared_histograms_cache: Optional[Dict[Path, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Args:
            data_dir: Base directory containing model directories
            min_context_steps: Minimum number of context observations
            max_query_gap: Maximum gap ahead to predict
            split: "train" or "val"
            train_ratio: Ratio of data for training
            seed: Random seed for splitting
            drop_prob: Probability of dropping intermediate context accuracies
            target_list: If specified, only predict these tasks
            heldout_list: Task names to exclude from loading
            num_bins: Number of histogram bins (must match histograms_by_step.npz)
            inverse_perplexity: If True, load histograms_by_step_invp.npz
            _shared_samples: Internal parameter to share pre-built samples between train/val
            _shared_histograms_cache: Internal parameter to share histograms cache between train/val
        """
        self.target_list = target_list
        self.heldout_list = heldout_list if heldout_list is not None else ["avg_losses_by_step", "histograms_by_step"]
        self.num_bins = num_bins
        # Share histograms cache if provided (for val dataset), otherwise create empty
        self._histograms_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = _shared_histograms_cache if _shared_histograms_cache is not None else {}
        self._task_files_cache = {}

        super().__init__(
            data_dir=data_dir,
            min_context_steps=min_context_steps,
            max_query_gap=max_query_gap,
            max_encoder_tokens=max_encoder_tokens,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            drop_prob=drop_prob,
            loss_file_pattern=loss_file_pattern,
            max_clip=max_clip,
            inverse_perplexity=inverse_perplexity,
            _shared_samples=_shared_samples,
        )

    def _get_task_files(self, model_dir: Path) -> Dict[str, Path]:
        """Get task .npy files (cached)."""
        if model_dir not in self._task_files_cache:
            task_files = {}
            for npy_file in model_dir.glob("*.npy"):
                task_name = npy_file.stem
                if self.target_list is not None:
                    if task_name not in self.target_list:
                        continue
                else:
                    if task_name in self.heldout_list:
                        continue
                task_files[task_name] = npy_file
            self._task_files_cache[model_dir] = task_files
        return self._task_files_cache[model_dir]

    def _build_sample_index(self) -> List[Dict]:
        """Build sample index using histograms_by_step.npz."""
        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Check that histogram file exists
            hist_filename = "histograms_by_step_invp.npz" if self.inverse_perplexity else "histograms_by_step.npz"
            hist_file = model_dir / hist_filename
            if not hist_file.exists():
                logging.warning(f"Missing {hist_filename} for {model_dir.name}")
                continue

            # Load histograms
            hist_data = np.load(hist_file)
            hist_steps = hist_data['steps'].astype(np.int64)
            histograms = hist_data['histograms'].astype(np.float32)  # (num_steps, num_bins)
            self._histograms_cache[model_dir] = (hist_steps, histograms)

            # Create step-to-index mapping for histograms
            hist_step_to_idx = {s: i for i, s in enumerate(hist_steps)}

            # Get task files
            task_files = self._get_task_files(model_dir)
            if not task_files:
                continue

            # Load task data and find common steps (intersection with histogram steps)
            tasks_to_process = list(task_files.keys())
            task_data = {}
            common_steps = set(hist_steps)

            for task_name in tasks_to_process:
                data = np.load(task_files[task_name])
                steps = data[0].astype(np.int64)
                accuracies = data[1]
                task_data[task_name] = {"steps": steps, "accuracies": accuracies}
                common_steps &= set(steps)

            if len(common_steps) < self.min_context_steps + 1:
                continue

            common_steps = sorted(common_steps)

            # Filter out step 0 if requested
            if self.exclude_step0:
                common_steps = [s for s in common_steps if s != 0]

            if len(common_steps) < self.min_context_steps + 1:
                continue

            # Remap task accuracies to common steps
            for task_name, data in task_data.items():
                step_to_idx = {s: i for i, s in enumerate(data["steps"])}
                common_accuracies = np.array([
                    data["accuracies"][step_to_idx[s]] for s in common_steps
                ])
                common_accuracies = np.round(common_accuracies, decimals=3)
                task_data[task_name]["common_accuracies"] = common_accuracies

            # Create samples
            for task_name in tasks_to_process:
                accuracies = task_data[task_name]["common_accuracies"]

                for context_end_idx in range(self.min_context_steps - 1, len(common_steps) - 1):
                    context_step = common_steps[context_end_idx]

                    # Check histogram exists for this step
                    if context_step not in hist_step_to_idx:
                        continue

                    max_target_idx = min(context_end_idx + self.max_query_gap + 1, len(common_steps))

                    for target_idx in range(context_end_idx + 1, max_target_idx):
                        target_step = common_steps[target_idx]

                        # Check histogram exists for target step
                        if target_step not in hist_step_to_idx:
                            continue

                        query_gap = target_idx - context_end_idx

                        samples.append({
                            "model_dir": model_dir,
                            "model_name": model_dir.name,
                            "task_name": task_name,
                            "context_end_idx": context_end_idx,
                            "target_idx": target_idx,
                            "query_gap": query_gap,
                            "steps": np.array(common_steps),
                            "values": accuracies,
                            # Store histogram indices for delta computation
                            "context_hist_idx": hist_step_to_idx[context_step],
                            "target_hist_idx": hist_step_to_idx[target_step],
                        })

        logging.info(f"Found {len(samples)} samples for MetaAccuracyDatasetDeltaHist")
        return samples

    def _load_encoder_input(self, sample: Dict) -> np.ndarray:
        """Load delta histogram (target - context_end)."""
        model_dir = sample["model_dir"]
        hist_steps, histograms = self._histograms_cache[model_dir]

        context_hist = histograms[sample["context_hist_idx"]]
        target_hist = histograms[sample["target_hist_idx"]]

        # Compute query-relative delta: target - context_end
        delta_histogram = target_hist - context_hist
        return delta_histogram.astype(np.float32)

    def _get_context_and_target(self, sample: Dict) -> Tuple[np.ndarray, float, float]:
        """Get context accuracies and target accuracy."""
        context_end_idx = sample["context_end_idx"]
        target_idx = sample["target_idx"]
        values = sample["values"]

        context_values = values[:context_end_idx + 1].copy()
        target_value = values[target_idx]

        context_values = np.round(context_values, decimals=3)
        target_value = round(float(target_value), 3)

        return context_values.astype(np.float32), target_value, 1.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)
        result["task_name"] = self.samples[idx]["task_name"]
        return result


class MetaAccuracyDatasetAvgLoss(MetaDataset):
    """
    Dataset for accuracy prediction using average losses as soft prompt input.

    Instead of raw token losses, this dataset provides:
    - encoder_input: All average losses from step 0 to context_end step

    The avg_losses_by_step.npy file must exist for each model directory.
    """

    def __init__(
        self,
        data_dir: str,
        min_context_steps: int = 4,
        max_query_gap: int = 10,
        max_encoder_tokens: int = 8192,  # Not used but kept for API compatibility
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        drop_prob: float = 0.0,
        target_list: Optional[List[str]] = None,
        heldout_list: Optional[List[str]] = None,
        loss_file_pattern: str = "word_losses*.npy",  # Not used
        max_clip: float = 20.0,  # Not used
        _shared_samples: Optional[List[Dict]] = None,
        _shared_avg_losses_cache: Optional[Dict] = None,
    ):
        self.target_list = target_list
        self.heldout_list = heldout_list if heldout_list is not None else ["avg_losses_by_step", "histograms_by_step"]
        self._task_files_cache = {}
        # Share avg_losses_cache if provided (for val dataset), otherwise create empty
        self._avg_losses_cache = _shared_avg_losses_cache if _shared_avg_losses_cache is not None else {}
        super().__init__(
            data_dir=data_dir,
            min_context_steps=min_context_steps,
            max_query_gap=max_query_gap,
            max_encoder_tokens=max_encoder_tokens,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            drop_prob=drop_prob,
            loss_file_pattern=loss_file_pattern,
            max_clip=max_clip,
            _shared_samples=_shared_samples,
        )

    def _get_task_files(self, model_dir: Path) -> Dict[str, Path]:
        """Get task .npy files (cached)."""
        if model_dir not in self._task_files_cache:
            task_files = {}
            for npy_file in model_dir.glob("*.npy"):
                task_name = npy_file.stem
                if self.target_list is not None:
                    if task_name not in self.target_list:
                        continue
                else:
                    if task_name in self.heldout_list:
                        continue
                task_files[task_name] = npy_file
            self._task_files_cache[model_dir] = task_files
        return self._task_files_cache[model_dir]

    def _build_sample_index(self) -> List[Dict]:
        """Build sample index using avg_losses_by_step.npy."""
        from src.meta.utils import _extract_step_number

        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Check that avg_losses_by_step.npy exists
            avg_loss_file = model_dir / "avg_losses_by_step.npy"
            if not avg_loss_file.exists():
                logging.warning(f"Missing avg_losses_by_step.npy for {model_dir.name}")
                continue

            # Load avg losses
            avg_data = np.load(avg_loss_file)
            avg_steps = avg_data[0].astype(np.int64)
            avg_losses = avg_data[1].astype(np.float32)
            self._avg_losses_cache[model_dir] = (avg_steps, avg_losses)

            # Get task files
            task_files = self._get_task_files(model_dir)
            if not task_files:
                continue

            # Load task data and find common steps
            tasks_to_process = list(task_files.keys())
            task_data = {}
            common_steps = set(avg_steps)

            for task_name in tasks_to_process:
                data = np.load(task_files[task_name])
                steps = data[0].astype(np.int64)
                accuracies = data[1]
                task_data[task_name] = {"steps": steps, "accuracies": accuracies}
                common_steps &= set(steps)

            if len(common_steps) < self.min_context_steps + 1:
                continue

            common_steps = sorted(common_steps)

            # Filter out step 0 if requested
            if self.exclude_step0:
                common_steps = [s for s in common_steps if s != 0]

            if len(common_steps) < self.min_context_steps + 1:
                continue

            # Create step-to-index mapping for avg losses
            avg_step_to_idx = {s: i for i, s in enumerate(avg_steps)}

            # Remap task accuracies to common steps
            for task_name, data in task_data.items():
                step_to_idx = {s: i for i, s in enumerate(data["steps"])}
                common_accuracies = np.array([
                    data["accuracies"][step_to_idx[s]] for s in common_steps
                ])
                common_accuracies = np.round(common_accuracies, decimals=3)
                task_data[task_name]["common_accuracies"] = common_accuracies

            # Create samples
            for task_name in tasks_to_process:
                accuracies = task_data[task_name]["common_accuracies"]

                for context_end_idx in range(self.min_context_steps - 1, len(common_steps) - 1):
                    context_step = common_steps[context_end_idx]

                    # Check avg loss exists for this step
                    if context_step not in avg_step_to_idx:
                        continue

                    max_target_idx = min(context_end_idx + self.max_query_gap + 1, len(common_steps))

                    for target_idx in range(context_end_idx + 1, max_target_idx):
                        query_gap = target_idx - context_end_idx

                        # Get avg loss indices for all steps up to context_end
                        avg_loss_indices = [avg_step_to_idx[common_steps[i]] for i in range(context_end_idx + 1)
                                            if common_steps[i] in avg_step_to_idx]

                        samples.append({
                            "model_dir": model_dir,
                            "model_name": model_dir.name,
                            "task_name": task_name,
                            "context_end_idx": context_end_idx,
                            "target_idx": target_idx,
                            "query_gap": query_gap,
                            "steps": np.array(common_steps),
                            "values": accuracies,
                            "avg_loss_indices": avg_loss_indices,
                        })

        logging.info(f"Found {len(samples)} samples for MetaAccuracyDatasetAvgLoss")
        return samples

    def _load_encoder_input(self, sample: Dict) -> np.ndarray:
        """Load average losses up to context_end step."""
        model_dir = sample["model_dir"]
        avg_steps, avg_losses = self._avg_losses_cache[model_dir]
        indices = sample["avg_loss_indices"]
        return avg_losses[indices].astype(np.float32)

    def _get_context_and_target(self, sample: Dict) -> Tuple[np.ndarray, float, float]:
        """Get context accuracies and target accuracy."""
        context_end_idx = sample["context_end_idx"]
        target_idx = sample["target_idx"]
        values = sample["values"]

        context_values = values[:context_end_idx + 1].copy()
        target_value = values[target_idx]

        context_values = np.round(context_values, decimals=3)
        target_value = round(float(target_value), 3)

        return context_values.astype(np.float32), target_value, 1.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)
        result["task_name"] = self.samples[idx]["task_name"]
        return result
