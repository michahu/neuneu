"""
Datasets for training neural scaling law predictors.

This package provides:
- MetaDataset: Abstract base class for temporal meta-prediction
- MetaLossDataset: Predicts loss values from temporal context
- MetaAccuracyDataset: Predicts task accuracies from temporal context
- MetaTaskAccuracyDataset: Diagnostic dataset with task accuracies as encoder input
- MetaLossEvalDataset: Evaluation dataset for comprehensive loss prediction
- MetaAccuracyDatasetAvgLoss: Uses average losses as encoder input
- ProbeDataset: Single-step loss → accuracy mapping
- DeltaProbeDataset: Delta histogram → delta accuracy mapping
- Various collate functions and factory functions
"""

from src.meta.datasets.base import (
    MetaDataset,
    preload_loss_files,
    get_cached_losses,
    load_task_accuracies,
    compute_chance_offsets,
)

from src.meta.datasets.meta import (
    MetaLossDataset,
    MetaAccuracyDataset,
    MetaTaskAccuracyDataset,
    MetaLossEvalDataset,
    MetaAccuracyDatasetAvgLoss,
    MetaAccuracyDatasetDeltaHist,
    collate_fn,
    collate_fn_accuracy,
)

from src.meta.datasets.probe import (
    ProbeDataset,
    DeltaProbeDataset,
    probe_collate_fn,
    delta_probe_collate_fn,
    create_probe_dataloaders,
    create_delta_probe_dataloaders,
)

__all__ = [
    # Base
    "MetaDataset",
    "preload_loss_files",
    "get_cached_losses",
    "load_task_accuracies",
    "compute_chance_offsets",
    # Meta
    "MetaLossDataset",
    "MetaAccuracyDataset",
    "MetaTaskAccuracyDataset",
    "MetaLossEvalDataset",
    "MetaAccuracyDatasetAvgLoss",
    "MetaAccuracyDatasetDeltaHist",
    "collate_fn",
    "collate_fn_accuracy",
    # Probe
    "ProbeDataset",
    "DeltaProbeDataset",
    "probe_collate_fn",
    "delta_probe_collate_fn",
    "create_probe_dataloaders",
    "create_delta_probe_dataloaders",
]
