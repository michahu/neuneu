"""
Probe datasets for direct mapping from losses to accuracies.

This module provides:
- ProbeDataset: Single-step loss → accuracy mapping
- DeltaProbeDataset: Delta histogram → delta accuracy mapping
- probe_collate_fn: Collate function for ProbeDataset
- delta_probe_collate_fn: Collate function for DeltaProbeDataset
- create_probe_dataloaders: Factory function for probe dataloaders
- create_delta_probe_dataloaders: Factory function for delta probe dataloaders
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.meta.datasets.base import preload_loss_files, get_cached_losses, load_task_accuracies


def load_model_histogram_data(
    model_dir: Path,
    tasks: List[str],
    exclude_step0: bool,
    inverse_perplexity: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[int, int], List[int], Dict[int, np.ndarray]]]:
    """
    Load histogram data and task accuracies for a single model directory.

    Args:
        model_dir: Path to the model directory
        tasks: List of task names to load
        exclude_step0: If True, exclude step 0 from valid steps
        inverse_perplexity: If True, load histograms_by_step_invp.npz instead of histograms_by_step.npz

    Returns:
        None if data is missing or invalid, otherwise a tuple of:
        - hist_steps: Array of step numbers with histograms
        - histograms: Array of histograms (num_steps, num_bins)
        - hist_step_to_idx: Mapping from step number to histogram index
        - valid_steps: List of valid step numbers (intersection of histogram and task steps)
        - step_to_task_accs: Mapping from step number to task accuracies array
    """
    # Check that histogram file exists
    hist_filename = "histograms_by_step_invp.npz" if inverse_perplexity else "histograms_by_step.npz"
    hist_file = model_dir / hist_filename
    if not hist_file.exists():
        logging.warning(f"Missing {hist_filename} for {model_dir.name}")
        return None

    # Load pre-computed histograms
    hist_data = np.load(hist_file)
    hist_steps = hist_data['steps'].astype(np.int64)
    histograms = hist_data['histograms'].astype(np.float32)  # (num_steps, num_bins)

    # Create step-to-index mapping for histograms
    hist_step_to_idx = {s: i for i, s in enumerate(hist_steps)}

    # Load task accuracies and find common steps (intersect with histogram steps)
    task_data, valid_steps = load_task_accuracies(
        model_dir, tasks,
        exclude_step0=exclude_step0,
        required_steps=set(hist_steps),
    )

    if not task_data or len(valid_steps) < 2:
        return None

    # Pre-compute step index to accuracy mapping for efficiency
    step_to_task_accs = {}
    for step in valid_steps:
        task_accs = []
        valid = True
        for task in tasks:
            if task in task_data:
                steps = task_data[task]["steps"]
                accs = task_data[task]["accuracies"]
                idx = np.where(steps == step)[0]
                if len(idx) > 0:
                    task_accs.append(accs[idx[0]])
                else:
                    valid = False
                    break
            else:
                valid = False
                break
        if valid and len(task_accs) == len(tasks):
            step_to_task_accs[step] = np.array(task_accs, dtype=np.float32)

    return hist_steps, histograms, hist_step_to_idx, valid_steps, step_to_task_accs


class ProbeDataset(Dataset):
    """
    Dataset for Direct Probe training - simplified version without temporal context.

    Each sample is a single (model, step) checkpoint that predicts task accuracies
    directly from token-level losses, without using previous accuracy observations.

    This is a simpler task than MetaAccuracyDataset: just map losses → accuracies
    without the temporal forecasting component.

    Output per sample:
    - losses: (seq_len,) token-level cross-entropy losses
    - histogram: (num_bins,) pre-computed histogram from histograms_by_step.npz
    - avg_accuracy: scalar average accuracy across tasks
    - task_accuracies: (num_tasks,) individual task accuracies
    - model_name, step: metadata
    """

    def __init__(
        self,
        data_dir: str,
        tasks: Optional[List[str]] = None,
        max_seq_len: int = 100000,
        seed: int = 42,
        split: str = "train",
        val_fraction: float = 0.15,
        loss_file_pattern: str = "word_losses*.npy",
        heldout_tasks: Optional[List[str]] = None,
        max_clip: float = 20.0,
        exclude_step0: bool = True,
        inverse_perplexity: bool = False,
        chance_accuracy_map: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            data_dir: Path to directory containing model subdirectories
            tasks: List of task names to include. If None, auto-discovers tasks.
            max_seq_len: Maximum sequence length (subsample if longer)
            seed: Random seed for subsampling and split
            split: "train" or "val"
            val_fraction: Fraction of data for validation
            loss_file_pattern: Glob pattern for loss files in step directories
            heldout_tasks: Tasks to exclude (default: ["avg_losses_by_step"])
            max_clip: Maximum loss value to clip to
            exclude_step0: If True, exclude step 0 from datasets (default: True)
            inverse_perplexity: If True, load histograms_by_step_invp.npz instead of histograms_by_step.npz
            chance_accuracy_map: Custom mapping of task names to chance accuracies.
        """
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.split = split
        self.val_fraction = val_fraction
        self.loss_file_pattern = loss_file_pattern
        self.heldout_tasks = heldout_tasks if heldout_tasks is not None else ["avg_losses_by_step"]
        self.max_clip = max_clip
        self.exclude_step0 = exclude_step0
        self.inverse_perplexity = inverse_perplexity
        self.chance_accuracy_map = chance_accuracy_map

        self.rng = np.random.RandomState(seed)

        # Cache for pre-computed histograms per model
        self._histograms_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}

        # Discover or validate tasks
        self.tasks = self._resolve_tasks(tasks)

        # Load all samples
        self.samples = self._load_samples()

        # Split into train/val
        self._apply_split()

        # Preload loss files
        self._preload_loss_files()

        logging.info(f"ProbeDataset ({split}): {len(self.samples)} samples, {len(self.tasks)} tasks")

    def _resolve_tasks(self, requested_tasks: Optional[List[str]]) -> List[str]:
        """Resolve which tasks to use."""
        # Find first model directory (skip hidden directories like .cache)
        model_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not model_dirs:
            raise ValueError(f"No model directories found in {self.data_dir}")

        # Discover available tasks from first model
        first_model = model_dirs[0]
        available_tasks = set()
        for npy_file in first_model.glob("*.npy"):
            task_name = npy_file.stem
            if task_name not in self.heldout_tasks:
                available_tasks.add(task_name)

        if not available_tasks:
            raise ValueError(f"No task .npy files found in {first_model}")

        if requested_tasks is not None:
            missing = set(requested_tasks) - available_tasks
            if missing:
                logging.warning(f"Requested tasks not found: {missing}. Available: {sorted(available_tasks)}")
            tasks = [t for t in requested_tasks if t in available_tasks]
            if not tasks:
                raise ValueError(f"None of requested tasks found. Available: {sorted(available_tasks)}")
            return tasks
        else:
            return sorted(available_tasks)

    def _load_samples(self) -> List[Dict]:
        """Load all (model, step) samples with their accuracies and histograms."""
        from src.meta.utils import _extract_step_number

        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue

            # Load histogram data and task accuracies using shared function
            result = load_model_histogram_data(
                model_dir, self.tasks, self.exclude_step0,
                inverse_perplexity=self.inverse_perplexity,
            )
            if result is None:
                continue

            hist_steps, histograms, hist_step_to_idx, valid_steps, step_to_task_accs = result
            self._histograms_cache[model_dir] = (hist_steps, histograms)

            # Load avg_losses_by_step.npy for this model (for avg_loss probe)
            avg_loss_file = model_dir / "avg_losses_by_step.npy"
            assert avg_loss_file.exists(), f"Missing avg_losses_by_step.npy in {model_dir}"
            avg_loss_data = np.load(avg_loss_file)
            avg_loss_steps = avg_loss_data[0].astype(int)
            avg_loss_values = avg_loss_data[1].astype(np.float32)
            step_to_avg_loss = {int(s): float(v) for s, v in zip(avg_loss_steps, avg_loss_values)}

            # Find which steps have loss files
            step_to_loss_files = {}
            for item in model_dir.iterdir():
                if not item.is_dir():
                    continue
                step_num = _extract_step_number(item.name)
                if step_num is None:
                    continue
                loss_files = list(item.glob(self.loss_file_pattern))
                if loss_files:
                    step_to_loss_files[step_num] = loss_files

            # Create samples for steps that have accuracies, histograms, and losses
            for step in valid_steps:
                assert step in step_to_avg_loss, (
                    f"Step {step} not found in avg_losses_by_step.npy for model: {model_dir}"
                )
                task_accs = step_to_task_accs[step]
                avg_loss = step_to_avg_loss[step]
                samples.append({
                    "model_dir": model_dir,
                    "model_name": model_dir.name,
                    "step": step,
                    "loss_files": step_to_loss_files[step],
                    "hist_idx": hist_step_to_idx[step],
                    "task_accuracies": task_accs,
                    "avg_accuracy": task_accs.mean(),
                    "avg_loss": avg_loss,
                })

        # Sort for reproducibility
        samples.sort(key=lambda x: (x["model_name"], x["step"]))

        return samples

    def _apply_split(self):
        """Split samples into train/val/all."""
        if self.split == "all":
            # Use all samples, no split
            return

        n = len(self.samples)
        n_val = int(n * self.val_fraction)

        indices = np.arange(n)
        self.rng.shuffle(indices)

        if self.split == "val":
            self.samples = [self.samples[i] for i in indices[:n_val]]
        else:
            self.samples = [self.samples[i] for i in indices[n_val:]]

    def _preload_loss_files(self):
        """Preload all loss files into cache."""
        all_files = set()
        for sample in self.samples:
            for f in sample["loss_files"]:
                all_files.add(str(f))

        if all_files:
            logging.info(f"Preloading {len(all_files)} loss files for {self.split} set...")
            preload_loss_files(
                list(all_files),
                max_tokens=self.max_seq_len,
                max_clip=self.max_clip,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load histogram from cache
        model_dir = sample["model_dir"]
        hist_steps, histograms = self._histograms_cache[model_dir]
        histogram = histograms[sample["hist_idx"]]

        # Load losses from cache or disk
        all_losses = []
        for loss_file in sample["loss_files"]:
            cached = get_cached_losses(str(loss_file))
            if cached is not None:
                all_losses.append(cached)
            else:
                # Fallback to disk
                losses = np.load(loss_file).flatten()
                losses = np.nan_to_num(losses, nan=0.0, posinf=10.0, neginf=0.0)
                losses = np.clip(losses, 0, self.max_clip)
                if len(losses) > self.max_seq_len:
                    losses = losses[:self.max_seq_len]
                all_losses.append(losses.astype(np.float32))

        losses = np.concatenate(all_losses) if len(all_losses) > 1 else all_losses[0]

        # Subsample if too long (deterministic based on idx)
        if len(losses) > self.max_seq_len:
            rng = np.random.RandomState(self.seed + idx)
            indices = rng.choice(len(losses), self.max_seq_len, replace=False)
            indices.sort()
            losses = losses[indices]

        task_accuracies = sample["task_accuracies"].copy()
        avg_accuracy = sample["avg_accuracy"]
        result = {
            "losses": torch.from_numpy(losses.copy()),
            "histogram": torch.from_numpy(histogram.copy()),
            "avg_accuracy": torch.tensor(avg_accuracy, dtype=torch.float32),
            "task_accuracies": torch.from_numpy(task_accuracies),
            "model_name": sample["model_name"],
            "step": sample["step"],
        }

        # Add avg_loss if available (for avg_loss probe)
        if sample.get("avg_loss") is not None:
            result["avg_loss"] = torch.tensor(sample["avg_loss"], dtype=torch.float32)

        return result

def probe_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for ProbeDataset"""
    result = {
        "losses": torch.stack([s["losses"] for s in batch]),
        "histogram": torch.stack([s["histogram"] for s in batch]),
        "avg_accuracy": torch.stack([s["avg_accuracy"] for s in batch]),
        "task_accuracies": torch.stack([s["task_accuracies"] for s in batch]),
        "model_names": [s["model_name"] for s in batch],
        "steps": [s["step"] for s in batch],
        "avg_loss": torch.stack([s["avg_loss"] for s in batch])
    }
    return result


def create_probe_dataloaders(
    data_dir: str = "results/datadecide_eval",
    tasks: Optional[List[str]] = None,
    max_seq_len: int = 100000,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    val_fraction: float = 0.15,
    loss_file_pattern: str = "word_losses*.npy",
    heldout_tasks: Optional[List[str]] = None,
    inverse_perplexity: bool = False,
    chance_accuracy_map: Optional[Dict[str, float]] = None,
) -> Tuple["DataLoader", "DataLoader"]:
    """Create train and validation dataloaders for ProbeDataset."""
    from torch.utils.data import DataLoader

    train_dataset = ProbeDataset(
        data_dir=data_dir,
        tasks=tasks,
        max_seq_len=max_seq_len,
        seed=seed,
        split="train",
        val_fraction=val_fraction,
        loss_file_pattern=loss_file_pattern,
        heldout_tasks=heldout_tasks,
        inverse_perplexity=inverse_perplexity,
        chance_accuracy_map=chance_accuracy_map,
    )

    val_dataset = ProbeDataset(
        data_dir=data_dir,
        tasks=tasks,
        max_seq_len=max_seq_len,
        seed=seed,
        split="val",
        val_fraction=val_fraction,
        loss_file_pattern=loss_file_pattern,
        heldout_tasks=heldout_tasks,
        inverse_perplexity=inverse_perplexity,
        chance_accuracy_map=chance_accuracy_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=probe_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=probe_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


class DeltaProbeDataset(Dataset):
    """
    Dataset for Delta Probe training - predicts change in accuracy from change in histogram.

    Each sample consists of a pair of checkpoints (step_i, step_j) where j > i:
    - delta_histogram: histogram(step_j) - histogram(step_i)
    - delta_accuracy: accuracy(step_j) - accuracy(step_i) for each task

    With data augmentation (max_gap > 1), we create pairs for all valid step combinations,
    not just consecutive steps. This dramatically expands training data:
    - With N steps and max_gap=1: N-1 pairs (consecutive only)
    - With N steps and max_gap=N: N*(N-1)/2 pairs (all combinations)

    This formulation learns the relationship between how the loss distribution
    evolves and how task performance changes, rather than trying to predict
    absolute accuracy from absolute histogram (which collapses to predicting mean).

    At inference time:
    1. Start from known (histogram_0, accuracy_0)
    2. For each future step, predict: accuracy_t = accuracy_{t-1} + model(delta_histogram)
    """

    def __init__(
        self,
        data_dir: str,
        tasks: Optional[List[str]] = None,
        seed: int = 42,
        split: str = "train",
        val_fraction: float = 0.15,
        loss_file_pattern: str = "word_losses*.npy",
        heldout_tasks: Optional[List[str]] = None,
        max_clip: float = 20.0,
        num_bins: int = 32,
        bin_min: float = 0.0,
        bin_max: float = 15.0,
        min_gap: int = 1,
        max_gap: Optional[int] = None,
        inverse_perplexity: bool = False,
        exclude_step0: bool = True,
        chance_accuracy_map: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            data_dir: Path to directory containing model subdirectories
            tasks: List of task names to include. If None, auto-discovers tasks.
            max_seq_len: Maximum sequence length for loss arrays
            seed: Random seed for split
            split: "train" or "val"
            val_fraction: Fraction of data for validation
            loss_file_pattern: Glob pattern for loss files in step directories
            heldout_tasks: Tasks to exclude (default: ["avg_losses_by_step"])
            max_clip: Maximum loss value to clip to
            num_bins: Number of histogram bins
            bin_min: Minimum loss value for binning
            bin_max: Maximum loss value for binning
            min_gap: Minimum gap between step indices (1 = consecutive only)
            max_gap: Maximum gap between step indices (None = unlimited, uses all pairs)
            inverse_perplexity: If True, transform losses x -> e^(-x) before binning
            exclude_step0: If True, exclude step 0 from datasets (default: True)
            chance_accuracy_map: Custom mapping of task names to chance accuracies.
        """
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.split = split
        self.val_fraction = val_fraction
        self.loss_file_pattern = loss_file_pattern
        self.heldout_tasks = heldout_tasks if heldout_tasks is not None else ["avg_losses_by_step"]
        self.max_clip = max_clip
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.inverse_perplexity = inverse_perplexity
        self.exclude_step0 = exclude_step0
        self.chance_accuracy_map = chance_accuracy_map

        self.rng = np.random.RandomState(seed)

        # Cache for pre-computed histograms per model
        self._histograms_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}

        # Discover or validate tasks
        self.tasks = self._resolve_tasks(tasks)

        # Load all samples (pairs with variable gaps)
        self.samples = self._load_samples()

        # Split into train/val
        self._apply_split()

        logging.info(f"DeltaProbeDataset ({split}): {len(self.samples)} samples, {len(self.tasks)} tasks, "
                     f"gap range: [{self.min_gap}, {self.max_gap or 'unlimited'}]")

    def _resolve_tasks(self, requested_tasks: Optional[List[str]]) -> List[str]:
        """Resolve which tasks to use."""
        model_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not model_dirs:
            raise ValueError(f"No model directories found in {self.data_dir}")

        first_model = model_dirs[0]
        available_tasks = set()
        for npy_file in first_model.glob("*.npy"):
            task_name = npy_file.stem
            if task_name not in self.heldout_tasks:
                available_tasks.add(task_name)

        if not available_tasks:
            raise ValueError(f"No task .npy files found in {first_model}")

        if requested_tasks is not None:
            missing = set(requested_tasks) - available_tasks
            if missing:
                logging.warning(f"Requested tasks not found: {missing}")
            tasks = [t for t in requested_tasks if t in available_tasks]
            if not tasks:
                raise ValueError(f"None of requested tasks found")
            return tasks
        else:
            return sorted(available_tasks)

    def _load_samples(self) -> List[Dict]:
        """Load all (step_i, step_j) pairs where j > i and gap is in [min_gap, max_gap].

        Loads pre-computed histograms from histograms_by_step.npz for efficiency.
        """
        samples = []

        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue

            # Load histogram data and task accuracies using shared function
            result = load_model_histogram_data(
                model_dir, self.tasks, self.exclude_step0,
                inverse_perplexity=self.inverse_perplexity,
            )
            if result is None:
                continue

            hist_steps, histograms, hist_step_to_idx, valid_steps, step_to_task_accs = result
            self._histograms_cache[model_dir] = (hist_steps, histograms)

            # Create samples for all valid step pairs within gap range
            # Gap is measured in step indices, not actual step values
            n_steps = len(valid_steps)
            effective_max_gap = self.max_gap if self.max_gap is not None else n_steps

            for i in range(n_steps):
                step_prev = valid_steps[i]
                if step_prev not in step_to_task_accs:
                    continue

                for gap in range(self.min_gap, min(effective_max_gap + 1, n_steps - i)):
                    j = i + gap
                    if j >= n_steps:
                        break

                    step_curr = valid_steps[j]
                    if step_curr not in step_to_task_accs:
                        continue

                    samples.append({
                        "model_dir": model_dir,
                        "model_name": model_dir.name,
                        "step_prev": step_prev,
                        "step_curr": step_curr,
                        "step_idx_prev": i,
                        "step_idx_curr": j,
                        "step_gap": gap,
                        "hist_idx_prev": hist_step_to_idx[step_prev],
                        "hist_idx_curr": hist_step_to_idx[step_curr],
                        "task_accuracies_prev": step_to_task_accs[step_prev],
                        "task_accuracies_curr": step_to_task_accs[step_curr],
                    })

        # Sort for reproducibility
        samples.sort(key=lambda x: (x["model_name"], x["step_prev"], x["step_gap"]))
        return samples

    def _apply_split(self):
        """Split samples into train/val."""
        if self.split == "all":
            return

        n = len(self.samples)
        n_val = int(n * self.val_fraction)

        indices = np.arange(n)
        self.rng.shuffle(indices)

        if self.split == "val":
            self.samples = [self.samples[i] for i in indices[:n_val]]
        else:
            self.samples = [self.samples[i] for i in indices[n_val:]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load histograms from cache
        model_dir = sample["model_dir"]
        hist_steps, histograms = self._histograms_cache[model_dir]
        histogram_prev = histograms[sample["hist_idx_prev"]]
        histogram_curr = histograms[sample["hist_idx_curr"]]

        # Compute delta histogram and delta accuracy
        delta_histogram = histogram_curr - histogram_prev
        delta_accuracy = sample["task_accuracies_curr"] - sample["task_accuracies_prev"]
        avg_delta_accuracy = delta_accuracy.mean()

        return {
            "delta_histogram": torch.from_numpy(delta_histogram.copy()),
            "delta_accuracy": torch.from_numpy(delta_accuracy.copy()),
            "avg_delta_accuracy": torch.tensor(avg_delta_accuracy, dtype=torch.float32),
            "histogram_prev": torch.from_numpy(histogram_prev.copy()),
            "histogram_curr": torch.from_numpy(histogram_curr.copy()),
            "accuracy_prev": torch.from_numpy(sample["task_accuracies_prev"].copy()),
            "accuracy_curr": torch.from_numpy(sample["task_accuracies_curr"].copy()),
            "model_name": sample["model_name"],
            "step_prev": sample["step_prev"],
            "step_curr": sample["step_curr"],
            "step_gap": sample["step_gap"],
        }


def delta_probe_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DeltaProbeDataset."""
    return {
        "delta_histogram": torch.stack([s["delta_histogram"] for s in batch]),
        "delta_accuracy": torch.stack([s["delta_accuracy"] for s in batch]),
        "avg_delta_accuracy": torch.stack([s["avg_delta_accuracy"] for s in batch]),
        "histogram_prev": torch.stack([s["histogram_prev"] for s in batch]),
        "histogram_curr": torch.stack([s["histogram_curr"] for s in batch]),
        "accuracy_prev": torch.stack([s["accuracy_prev"] for s in batch]),
        "accuracy_curr": torch.stack([s["accuracy_curr"] for s in batch]),
        "model_names": [s["model_name"] for s in batch],
        "steps_prev": [s["step_prev"] for s in batch],
        "steps_curr": [s["step_curr"] for s in batch],
        "step_gaps": torch.tensor([s["step_gap"] for s in batch], dtype=torch.long),
    }


def create_delta_probe_dataloaders(
    data_dir: str = "results/datadecide_eval",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    val_fraction: float = 0.15,
    loss_file_pattern: str = "word_losses*.npy",
    heldout_tasks: Optional[List[str]] = None,
    num_bins: int = 32,
    bin_min: float = 0.0,
    bin_max: float = 15.0,
    min_gap: int = 1,
    max_gap: Optional[int] = None,
    inverse_perplexity: bool = False,
    chance_accuracy_map: Optional[Dict[str, float]] = None,
) -> Tuple["DataLoader", "DataLoader"]:
    """
    Create train and validation dataloaders for DeltaProbeDataset.

    Args:
        data_dir: Path to directory containing model subdirectories
        tasks: List of task names to include
        max_seq_len: Maximum sequence length for loss arrays
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        seed: Random seed
        val_fraction: Fraction of data for validation
        loss_file_pattern: Glob pattern for loss files
        heldout_tasks: Tasks to exclude
        num_bins: Number of histogram bins
        bin_min: Minimum loss value for binning
        bin_max: Maximum loss value for binning
        min_gap: Minimum gap between step indices (1 = consecutive only)
        max_gap: Maximum gap between step indices (None = unlimited)
        inverse_perplexity: If True, transform losses x -> e^(-x) before binning
        chance_accuracy_map: Custom mapping of task names to chance accuracies

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = DeltaProbeDataset(
        data_dir=data_dir,
        tasks=tasks,
        seed=seed,
        split="train",
        val_fraction=val_fraction,
        loss_file_pattern=loss_file_pattern,
        heldout_tasks=heldout_tasks,
        num_bins=num_bins,
        bin_min=bin_min,
        bin_max=bin_max,
        min_gap=min_gap,
        max_gap=max_gap,
        inverse_perplexity=inverse_perplexity,
        chance_accuracy_map=chance_accuracy_map,
    )

    val_dataset = DeltaProbeDataset(
        data_dir=data_dir,
        tasks=tasks,
        seed=seed,
        split="val",
        val_fraction=val_fraction,
        loss_file_pattern=loss_file_pattern,
        heldout_tasks=heldout_tasks,
        num_bins=num_bins,
        bin_min=bin_min,
        bin_max=bin_max,
        min_gap=min_gap,
        max_gap=max_gap,
        inverse_perplexity=inverse_perplexity,
        chance_accuracy_map=chance_accuracy_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=delta_probe_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=delta_probe_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
