"""
Base classes and utilities for meta-prediction datasets.

This module provides:
- Caching utilities for efficient data loading
- MetaDataset abstract base class for temporal meta-prediction
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Global cache for preloaded loss arrays (shared across workers via fork)
# Key: file path string, Value: preprocessed numpy array (float32, clipped)
_LOSS_ARRAY_CACHE: Dict[str, np.ndarray] = {}


def preload_loss_files(
    file_paths: List[str],
    max_tokens: int = 8192,
    max_clip: float = 20.0,
    inverse_perplexity: bool = False,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
) -> None:
    """
    Preload and preprocess all loss files into memory.

    This should be called before creating DataLoader workers to ensure
    the data is shared via copy-on-write after fork.

    Args:
        file_paths: List of paths to .npy loss files
        max_tokens: Maximum number of tokens to keep (truncate to this)
        max_clip: Maximum loss value to clip to (default 20 for token losses,
                  use higher value like 100 for word losses which are sums)
        inverse_perplexity: If True, transform losses x -> e^(-x) (values become 0-1)
        bin_min: Minimum value for clipping (0.0 for inverse_perplexity)
        bin_max: Maximum value for clipping (1.0 for inverse_perplexity, 15.0 otherwise)
    """
    global _LOSS_ARRAY_CACHE

    loaded_count = 0
    for path in file_paths:
        path_str = str(path)
        if path_str not in _LOSS_ARRAY_CACHE:
            losses = np.load(path_str)
            losses = losses.flatten()
            losses = np.nan_to_num(losses, nan=0.0, posinf=10.0, neginf=0.0)
            losses = np.clip(losses, 0, max_clip)

            # Apply inverse perplexity transform if enabled: x -> e^(-x)
            if inverse_perplexity:
                losses = np.exp(-losses)
                losses = np.clip(losses, bin_min, bin_max - 1e-6)

            if len(losses) > max_tokens:
                losses = losses[:max_tokens]
            _LOSS_ARRAY_CACHE[path_str] = losses.astype(np.float32)
            loaded_count += 1

    if loaded_count > 0:
        total_mb = sum(arr.nbytes for arr in _LOSS_ARRAY_CACHE.values()) / (1024 * 1024)
        logging.info(f"Preloaded {loaded_count} loss files ({total_mb:.1f} MB total in cache)")


def get_cached_losses(file_path: str) -> Optional[np.ndarray]:
    """Get preloaded losses from cache, or None if not cached."""
    return _LOSS_ARRAY_CACHE.get(str(file_path))


def load_task_accuracies(
    model_dir: Path,
    tasks: List[str],
    exclude_step0: bool = True,
    required_steps: Optional[set] = None,
) -> Tuple[Dict[str, Dict], List[int]]:
    """
    Load task accuracies for a model and find common steps.

    Args:
        model_dir: Path to model directory containing task .npy files
        tasks: List of task names to load
        exclude_step0: If True, exclude step 0 from valid steps
        required_steps: If provided, intersect with these steps (e.g., histogram steps)

    Returns:
        Tuple of:
        - task_data: Dict mapping task name to {"steps": array, "accuracies": array}
        - valid_steps: Sorted list of common steps across all tasks (and required_steps if provided)

    Raises:
        Returns empty dict and empty list if no valid data found.
    """
    task_data = {}
    for task in tasks:
        task_file = model_dir / f"{task}.npy"
        if task_file.exists():
            data = np.load(task_file)
            task_data[task] = {
                "steps": data[0].astype(int),
                "accuracies": data[1],
            }
        else:
            logging.debug(f"Task file not found: {task_file}")

    if not task_data:
        return {}, []

    # Find common steps across all loaded tasks
    common_steps = None
    for task in task_data:
        task_steps = set(task_data[task]["steps"])
        if common_steps is None:
            common_steps = task_steps
        else:
            common_steps &= task_steps

    if not common_steps:
        return task_data, []

    # Intersect with required_steps if provided
    if required_steps is not None:
        common_steps &= required_steps

    if not common_steps:
        return task_data, []

    # Get sorted list of valid steps
    valid_steps = sorted(common_steps)

    # Filter out step 0 if requested
    if exclude_step0:
        valid_steps = [s for s in valid_steps if s != 0]

    return task_data, valid_steps


def compute_chance_offsets(
    tasks: List[str],
    chance_accuracy_map: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute per-task chance accuracy offsets.

    Args:
        tasks: List of task names in order
        chance_accuracy_map: Custom mapping of task names to chance accuracies.
                             If None, uses TASK_CHANCE_ACCURACY from src.analysis.

    Returns:
        Array of shape (num_tasks,) with chance accuracy for each task.
    """
    from src.analysis import TASK_CHANCE_ACCURACY, DEFAULT_CHANCE_ACCURACY

    chance_map = chance_accuracy_map or TASK_CHANCE_ACCURACY
    offsets = []
    for task in tasks:
        chance = chance_map.get(task, DEFAULT_CHANCE_ACCURACY)
        offsets.append(chance)

    return np.array(offsets, dtype=np.float32)


class MetaDataset(Dataset, ABC):
    """
    Abstract base class for meta-prediction datasets.

    Each sample provides:
    - context_values: Observed values (losses or accuracies)
    - context_gaps: Gap before each observed value
    - query_gap: How far ahead to predict from the last context observation
    - target_value: The value to predict
    - encoder_input: Raw token-level losses for conditioning the soft prompt encoder
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
        loss_file_pattern: str = "word_losses*.npy",
        max_clip: float = 20.0,
        inverse_perplexity: bool = False,
        bin_min: float = 0.0,
        bin_max: float = 15.0,
        exclude_step0: bool = True,
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
            drop_prob: Probability of dropping intermediate context values (data augmentation)
            loss_file_pattern: Glob pattern for loss files
            max_clip: Maximum loss value to clip to (default 20 for token losses,
                     use 100 for word losses which are sums)
            inverse_perplexity: If True, transform losses x -> e^(-x) (values become 0-1)
            bin_min: Minimum value for clipping (0.0 for inverse_perplexity)
            bin_max: Maximum value for clipping (1.0 for inverse_perplexity, 15.0 otherwise)
            exclude_step0: If True, exclude step 0 from datasets (default: True)
            _shared_samples: Internal parameter to share pre-built samples between train/val
        """
        self.data_dir = Path(data_dir)
        self.min_context_steps = min_context_steps
        self.max_query_gap = max_query_gap
        self.max_encoder_tokens = max_encoder_tokens
        self.split = split
        self.drop_prob = drop_prob
        self.loss_file_pattern = loss_file_pattern
        self.max_clip = max_clip
        self.inverse_perplexity = inverse_perplexity
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.exclude_step0 = exclude_step0

        # Build index of all valid samples (or use shared samples)
        if _shared_samples is not None:
            all_samples = _shared_samples
        else:
            all_samples = self._build_sample_index()

        # Store full samples for potential sharing with val dataset
        self._all_samples = all_samples

        # Split into train/val
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            self.samples = [all_samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [all_samples[i] for i in indices[split_idx:]]

        logging.info(f"{split} set: {len(self.samples)} samples (drop_prob={drop_prob})")

        # Preload all loss files into memory (for efficient data loading)
        self._preload_loss_files()

    @abstractmethod
    def _build_sample_index(self) -> List[Dict]:
        """Build index of all valid samples. Implemented by subclasses."""
        pass

    def _preload_loss_files(self) -> None:
        """Preload all unique loss files used by this dataset's samples."""
        # Collect all unique loss file paths
        all_files = set()
        for sample in self.samples:
            if "raw_loss_files" in sample:
                for f in sample["raw_loss_files"]:
                    all_files.add(str(f))

        if all_files:
            logging.info(f"Preloading {len(all_files)} unique loss files for {self.split} set...")
            preload_loss_files(
                list(all_files),
                max_tokens=self.max_encoder_tokens,
                max_clip=self.max_clip,
                inverse_perplexity=self.inverse_perplexity,
                bin_min=self.bin_min,
                bin_max=self.bin_max,
            )

    def _load_encoder_input(self, sample: Dict) -> np.ndarray:
        """Load and concatenate raw losses from cache (or fallback to disk)."""
        all_losses = []
        for f in sample["raw_loss_files"]:
            # Try cache first (much faster)
            cached = get_cached_losses(str(f))
            if cached is not None:
                all_losses.append(cached)
            else:
                # Fallback to disk load (shouldn't happen after preloading)
                losses = np.load(f)
                losses = losses.flatten()
                losses = np.nan_to_num(losses, nan=0.0, posinf=10.0, neginf=0.0)
                losses = np.clip(losses, 0, self.max_clip)

                # Apply inverse perplexity transform if enabled: x -> e^(-x)
                if self.inverse_perplexity:
                    losses = np.exp(-losses)
                    losses = np.clip(losses, self.bin_min, self.bin_max - 1e-6)

                if len(losses) > self.max_encoder_tokens:
                    losses = losses[:self.max_encoder_tokens]
                all_losses.append(losses.astype(np.float32))

        if len(all_losses) == 1:
            return all_losses[0]

        losses = np.concatenate(all_losses)
        if len(losses) > self.max_encoder_tokens:
            losses = losses[:self.max_encoder_tokens]
        return losses

    @abstractmethod
    def _get_context_and_target(self, sample: Dict) -> Tuple[np.ndarray, float, float]:
        """
        Get context values and target for a sample.

        Returns:
            Tuple of (context_values, target_value, normalization_factor)
            normalization_factor is used to denormalize predictions (1.0 if no normalization)
        """
        pass

    def _apply_dropout(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random dropout to context values, keeping first and last."""
        n = len(values)
        if n <= 2:
            return values, np.arange(n)

        # Use torch.rand for proper RNG management with DataLoader workers
        keep_mask = torch.rand(n).numpy() >= self.drop_prob
        keep_mask[0] = True  # Always keep first
        keep_mask[-1] = True  # Always keep last

        kept_indices = np.where(keep_mask)[0]
        kept_values = values[keep_mask]
        return kept_values, kept_indices

    def _compute_gaps(self, indices: np.ndarray, target_idx: int) -> np.ndarray:
        """
        Compute forward gaps from indices.

        Each gap[i] represents the distance from indices[i] to indices[i+1],
        with the last gap being the distance from indices[-1] to target_idx.

        Args:
            indices: Array of context indices (positions of observed values)
            target_idx: Index of the target value to predict

        Returns:
            Array of gaps where gap[i] = distance from value[i] to value[i+1] (or target)
        """
        if len(indices) == 0:
            return np.array([], dtype=np.int64)

        gaps = np.zeros(len(indices), dtype=np.int64)
        # Gap from each value to the next value
        gaps[:-1] = np.diff(indices)
        # Last gap is from the last context value to the target
        gaps[-1] = target_idx - indices[-1]
        return gaps

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        target_idx = sample["target_idx"]

        # Get context and target (subclass-specific)
        context_values, target_value, norm_factor = self._get_context_and_target(sample)
        context_indices = np.arange(len(context_values))

        # Apply dropout for data augmentation (only during training)
        if self.drop_prob > 0 and self.split == "train":
            context_values, context_indices = self._apply_dropout(context_values)

        # Compute forward gaps (each gap[i] = distance from value[i] to value[i+1] or target)
        context_gaps = self._compute_gaps(context_indices, target_idx)

        # Load encoder input (subclass-specific)
        encoder_input = self._load_encoder_input(sample)

        # Use torch.from_numpy for zero-copy conversion where possible
        return {
            "encoder_input": torch.from_numpy(encoder_input.copy()),  # copy needed since cache is shared
            "context_values": torch.from_numpy(context_values),
            "context_gaps": torch.from_numpy(context_gaps),
            "target_value": torch.tensor(target_value, dtype=torch.float32),
            "norm_factor": torch.tensor(norm_factor, dtype=torch.float32),
            "model_name": sample["model_name"],
        }
