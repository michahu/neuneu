"""
Probes for predicting downstream task accuracy from token-level losses.

This module contains:
1. DirectProbe: CNN encoder that predicts task accuracy
2. HistogramProbe: High-bias model using loss histograms
3. DeltaHistogramProbe: Predicts accuracy changes from histogram changes
4. KLDeltaProbe: Uses KL divergence features for delta prediction
5. AverageLossProbe: MLP that predicts task accuracy from average loss (neural logistic equivalent)

Usage:
    from src.meta.probes import (
        DirectProbe,
        HistogramProbe,
        DeltaHistogramProbe,
        KLDeltaProbe,
        AverageLossProbe,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Literal

from src.meta.utils import pinball_loss
from src.meta.model import CNNSoftPromptGenerator


LossType = Literal["mse", "quantile"]

# Default quantiles for uncertainty estimation
DEFAULT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


# =============================================================================
# Utility functions
# =============================================================================

def compute_loss_linear(
    pred: torch.Tensor,
    targets: torch.Tensor,
    loss_type: LossType,
    quantiles: List[float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss for per-task predictions.

    Args:
        pred: Model predictions
            - loss_type="mse": (batch, num_tasks)
            - loss_type="quantile": (batch, num_tasks, num_quantiles)
        targets: (batch, num_tasks) per-task accuracy targets
        loss_type: "mse" or "quantile"
        quantiles: List of quantiles (for quantile loss)

    Returns:
        loss: Scalar loss
        metrics: Dictionary with loss metrics
    """
    if loss_type == "mse":
        # Per-task MSE, averaged
        loss = F.mse_loss(pred, targets)

        with torch.no_grad():
            mae = F.l1_loss(pred, targets).item()
            rmse = torch.sqrt(loss).item()

            # Per-task metrics
            task_mae = F.l1_loss(pred, targets, reduction='none').mean(dim=0)  # (num_tasks,)
            task_mse = F.mse_loss(pred, targets, reduction='none').mean(dim=0)  # (num_tasks,)

            # Correlation on flattened predictions
            pred_flat = pred.flatten()
            target_flat = targets.flatten()
            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()
            correlation = (pred_centered * target_centered).sum() / (
                pred_centered.norm() * target_centered.norm() + 1e-8
            )

        metrics = {
            "loss": loss.item(),
            "mse": loss.item(),
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation.item(),
            "task_mae_mean": task_mae.mean().item(),
            "task_mse_mean": task_mse.mean().item(),
        }
    else:  # quantile
        loss = pinball_loss(pred, targets, quantiles)

        with torch.no_grad():
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            median_pred = pred[:, :, median_idx]  # (batch, num_tasks)

            mae = F.l1_loss(median_pred, targets).item()
            mse = F.mse_loss(median_pred, targets).item()
            rmse = mse ** 0.5

            task_mae = F.l1_loss(median_pred, targets, reduction='none').mean(dim=0)

            # Calibration (on average across tasks)
            avg_pred = pred.mean(dim=1)  # (batch, num_quantiles)
            avg_target = targets.mean(dim=1)  # (batch,)
            calibration_50, calibration_80 = _compute_calibration(avg_pred, avg_target, quantiles)

        metrics = {
            "loss": loss.item(),
            "pinball": loss.item(),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "task_mae_mean": task_mae.mean().item(),
            "calibration_50": calibration_50,
            "calibration_80": calibration_80,
        }

    return loss, metrics


def _compute_calibration(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: List[float],
) -> Tuple[float, float]:
    """Compute calibration metrics for quantile predictions."""
    if 0.25 in quantiles and 0.75 in quantiles:
        q25_idx = quantiles.index(0.25)
        q75_idx = quantiles.index(0.75)
        in_50 = ((target >= pred[:, q25_idx]) &
                 (target <= pred[:, q75_idx])).float().mean().item()
    else:
        in_50 = float('nan')

    if 0.1 in quantiles and 0.9 in quantiles:
        q10_idx = quantiles.index(0.1)
        q90_idx = quantiles.index(0.9)
        in_80 = ((target >= pred[:, q10_idx]) &
                 (target <= pred[:, q90_idx])).float().mean().item()
    else:
        in_80 = float('nan')

    return in_50, in_80


# =============================================================================
# Direct Probe (CNN-based)
# =============================================================================


class DirectProbe(nn.Module):
    """
    Unified probe model supporting different head types and loss types.

    Uses CNNSoftPromptGenerator to encode raw loss values into features,
    then applies an MLP head for task prediction.

    Loss types:
        - "mse": Standard MSE loss
        - "quantile": Pinball loss for quantile regression
    """

    def __init__(
        self,
        loss_type: LossType = "mse",
        num_tasks: int = 5,
        quantiles: List[float] = None,
        input_len: int = 100000,
        channels: Tuple[int, ...] = (4, 8, 16, 32),
        kernel_size: int = 16,
        stride: int = 8,
        encoder_hidden_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.num_tasks = num_tasks
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.num_quantiles = len(self.quantiles)

        # Use CNNSoftPromptGenerator as encoder
        # It outputs (batch, 1, encoder_hidden_dim), we'll squeeze to (batch, encoder_hidden_dim)
        self.encoder = CNNSoftPromptGenerator(
            input_len=input_len,
            hidden_dim=encoder_hidden_dim,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Determine output dimension based on loss_type
        if loss_type == "quantile":
            output_dim = num_tasks * self.num_quantiles
        else:
            output_dim = num_tasks

        self._output_dim = output_dim

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(encoder_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            losses: (batch, seq_len) token-level loss values
        Returns:
            - loss_type="mse": (batch, num_tasks) per-task predictions
            - loss_type="quantile": (batch, num_tasks, num_quantiles) per-task quantile predictions
        """
        # CNNSoftPromptGenerator returns (batch, 1, hidden_dim), squeeze to (batch, hidden_dim)
        features = self.encoder(losses).squeeze(1)
        output = self.head(features)

        if self.loss_type == "mse":
            return output  # (batch, num_tasks)
        else:  # quantile
            return output.view(-1, self.num_tasks, self.num_quantiles)  # (batch, num_tasks, num_quantiles)

    def forward_median(self, losses: torch.Tensor) -> torch.Tensor:
        """Get median prediction (for quantile loss)."""
        if self.loss_type != "quantile":
            raise ValueError("forward_median only valid for quantile loss")

        output = self.forward(losses)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2


        return output[:, :, median_idx]  # (batch, num_tasks)

    def get_features(self, losses: torch.Tensor) -> torch.Tensor:
        """Get CNN features (for analysis)."""
        return self.encoder(losses).squeeze(1)

    def compute_loss(
        self,
        losses: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss based on loss_type.

        Args:
            losses: (batch, seq_len) token-level losses
            targets: (batch, num_tasks) per-task accuracies
        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss metrics
        """
        pred = self.forward(losses)
        return compute_loss_linear(pred, targets, self.loss_type, self.quantiles)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Histogram Probe
# =============================================================================


class HistogramProbe(nn.Module):
    """
    High-bias probe that predicts per-task accuracy from precomputed loss histograms.

    This model has much higher bias than the CNN probe because:
    1. It only sees distributional information (no positional patterns)
    2. The MLP is intentionally small
    3. The histogram representation is fixed and interpretable
    """

    def __init__(
        self,
        num_bins: int = 32,
        hidden_dims: Tuple[int, ...] = (64, 32),
        loss_type: LossType = "mse",
        num_tasks: int = 32,
        quantiles: List[float] = None,
        use_rms_norm: bool = False,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.loss_type = loss_type
        self.num_tasks = num_tasks
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.num_quantiles = len(self.quantiles)
        self.use_rms_norm = use_rms_norm

        # Determine output dimension per task
        if loss_type == "quantile":
            self._output_per_task = self.num_quantiles
        else:
            self._output_per_task = 1

        # Total output: num_tasks * output_per_task
        output_dim = num_tasks * self._output_per_task

        # Build MLP
        # Input dim is num_bins + 1 if using RMS norm (extra feature for magnitude)
        layers = []
        in_dim = num_bins + 1 if use_rms_norm else num_bins
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, histogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            histogram: (batch, num_bins) precomputed histogram
        Returns:
            - loss_type="mse": (batch, num_tasks) per-task predictions
            - loss_type="quantile": (batch, num_tasks, num_quantiles) per-task quantile predictions
        """
        if self.use_rms_norm:
            rms = histogram.pow(2).mean(dim=-1, keepdim=True).sqrt()
            histogram = torch.cat([histogram / (rms + 1e-8), rms], dim=-1)

        output = self.mlp(histogram)

        if self.loss_type == "mse":
            return output  # (batch, num_tasks)
        else:
            return output.view(-1, self.num_tasks, self.num_quantiles)  # (batch, num_tasks, num_quantiles)

    def forward_median(self, histogram: torch.Tensor) -> torch.Tensor:
        """Get median prediction (for quantile loss)."""
        if self.loss_type != "quantile":
            raise ValueError("forward_median only valid for quantile loss")
        output = self.forward(histogram)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return output[:, :, median_idx]  # (batch, num_tasks)

    def compute_loss(
        self,
        histogram: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            histogram: (batch, num_bins) precomputed histogram
            targets: (batch, num_tasks) per-task accuracy targets
        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss metrics
        """
        pred = self.forward(histogram)
        return compute_loss_linear(pred, targets, self.loss_type, self.quantiles)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Delta Histogram Probe
# =============================================================================

class DeltaHistogramProbe(nn.Module):
    """
    Predicts change in accuracy (delta_y) from change in histogram (delta_x).

    Instead of learning: histogram -> accuracy (which collapses to mean)
    We learn: (histogram_t - histogram_{t-1}) -> (accuracy_t - accuracy_{t-1})

    This formulation:
    1. Removes the need to learn absolute accuracy levels per task
    2. Focuses on the relationship between loss distribution changes and performance changes
    3. Should generalize better across tasks and models

    At inference time, we start from a known (histogram_0, accuracy_0) and
    iteratively predict: accuracy_t = accuracy_{t-1} + model(histogram_t - histogram_{t-1})

    Architecture:
    - RMSNorm normalizes delta histogram (preserves direction, loses some scale)
    - L2 norm of original delta is concatenated as extra feature (preserves scale info)
    - MLP predicts per-task delta accuracy from [RMSNorm(delta), L2_norm(delta)]
    """

    def __init__(
        self,
        num_bins: int = 32,
        hidden_dims: Tuple[int, ...] = (64, 32),
        loss_type: LossType = "mse",
        num_tasks: int = 32,
        quantiles: List[float] = None,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.loss_type = loss_type
        self.num_tasks = num_tasks
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.num_quantiles = len(self.quantiles)

        # Determine output dimension per task
        if loss_type == "quantile":
            self._output_per_task = self.num_quantiles
        else:
            self._output_per_task = 1

        # Total output: num_tasks * output_per_task
        output_dim = num_tasks * self._output_per_task

        # Simple MLP: delta_histogram -> delta_accuracy
        layers = []
        in_dim = num_bins
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, delta_histogram: torch.Tensor) -> torch.Tensor:
        """
        Predict delta_accuracy from delta_histogram.

        Args:
            delta_histogram: (batch, num_bins) difference between two precomputed histograms
        Returns:
            - loss_type="mse": (batch, num_tasks) predicted delta accuracy per task
            - loss_type="quantile": (batch, num_tasks, num_quantiles) quantile predictions per task
        """
        output = self.mlp(delta_histogram)

        if self.loss_type == "mse":
            return output  # (batch, num_tasks)
        else:
            return output.view(-1, self.num_tasks, self.num_quantiles)  # (batch, num_tasks, num_quantiles)

    def compute_loss(
        self,
        delta_histogram: torch.Tensor,
        delta_accuracy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for delta prediction.

        Args:
            delta_histogram: (batch, num_bins) histogram differences
            delta_accuracy: (batch, num_tasks) per-task accuracy differences (target)
        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss metrics
        """
        pred = self.forward(delta_histogram)
        return compute_loss_linear(pred, delta_accuracy, self.loss_type, self.quantiles)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# KL Delta Probe (KL Divergence-based)
# =============================================================================

class RMSPolarFeatures(nn.Module):
    """
    KL divergence-based feature extraction with RMS normalization.

    Instead of computing delta_histogram = hist_curr - hist_prev and normalizing,
    this computes pointwise KL divergence between the two histograms, applies
    symlog compression to handle explosions, and extracts direction + magnitude.

    Output: [normalized_direction (num_bins), rms_magnitude (1)]
    """

    def __init__(self, num_bins: int = 16, epsilon: float = 1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.epsilon = epsilon

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL-based features from two histograms.

        Args:
            p: (batch, num_bins) - current histogram (target distribution)
            q: (batch, num_bins) - previous histogram (reference distribution)

        Returns:
            (batch, num_bins + 1) - [normalized_direction, rms_magnitude]
        """
        # 1. Compute Pointwise KL (The Raw Signal)
        ratio = p / (q + self.epsilon)
        pointwise_kl = p * torch.log(ratio + self.epsilon)

        # 2. Compress (SymLog) to handle explosions
        # We do this BEFORE norm to prevent outliers from dominating the direction
        x = torch.asinh(pointwise_kl)

        # 3. Compute Magnitude (RMS)
        # This effectively calculates the "power" of the error signal
        rms_magnitude = x.pow(2).mean(dim=1, keepdim=True).sqrt()

        # 4. Normalize (The "Direction")
        # RMSNorm formula: x / RMS(x)
        normalized_direction = x / (rms_magnitude + self.epsilon)

        # 5. Concatenate
        # We feed the network:
        # A. The exact "shape" of the error (normalized)
        # B. The "intensity" of the error (rms_magnitude)
        return torch.cat([normalized_direction, rms_magnitude], dim=1)


class KLDeltaProbe(nn.Module):
    """
    Predicts change in accuracy (delta_y) using KL divergence features.

    Unlike DeltaHistogramProbe which uses delta_histogram = hist_curr - hist_prev,
    this model computes pointwise KL divergence between histograms and uses
    RMS normalization for feature extraction.

    This captures the "divergence" between distributions rather than simple
    arithmetic difference, which may be more meaningful for probability distributions.
    """

    def __init__(
        self,
        num_bins: int = 32,
        hidden_dims: Tuple[int, ...] = (64, 32),
        loss_type: LossType = "mse",
        num_tasks: int = 32,
        quantiles: List[float] = None,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.loss_type = loss_type
        self.num_tasks = num_tasks
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.num_quantiles = len(self.quantiles)

        # KL-based feature extraction: outputs [normalized_direction (num_bins), rms_magnitude (1)]
        self.kl_features = RMSPolarFeatures(num_bins=num_bins)

        # Determine output dimension per task
        if loss_type == "quantile":
            self._output_per_task = self.num_quantiles
        else:
            self._output_per_task = 1

        # Total output: num_tasks * output_per_task
        output_dim = num_tasks * self._output_per_task

        # Simple MLP: KL features (num_bins + 1) -> delta_accuracy
        layers = []
        in_dim = num_bins + 1
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hist_curr: torch.Tensor,
        hist_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict delta_accuracy from two precomputed histograms using KL features.

        Args:
            hist_curr: (batch, num_bins) current precomputed histogram
            hist_prev: (batch, num_bins) previous precomputed histogram

        Returns:
            - loss_type="mse": (batch, num_tasks) predicted delta accuracy per task
            - loss_type="quantile": (batch, num_tasks, num_quantiles) quantile predictions per task
        """
        # Extract KL-based features: [normalized_direction (num_bins), rms_magnitude (1)]
        features = self.kl_features(hist_curr, hist_prev)
        output = self.mlp(features)

        if self.loss_type == "mse":
            return output  # (batch, num_tasks)
        else:
            return output.view(-1, self.num_tasks, self.num_quantiles)  # (batch, num_tasks, num_quantiles)

    def compute_loss(
        self,
        hist_curr: torch.Tensor,
        hist_prev: torch.Tensor,
        delta_accuracy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for KL delta prediction.

        Args:
            hist_curr: (batch, num_bins) current histogram
            hist_prev: (batch, num_bins) previous histogram
            delta_accuracy: (batch, num_tasks) per-task accuracy differences (target)

        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss metrics
        """
        pred = self.forward(hist_curr, hist_prev)
        return compute_loss_linear(pred, delta_accuracy, self.loss_type, self.quantiles)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Average Loss Probe (Neural Logistic Equivalent)
# =============================================================================


class AverageLossProbe(nn.Module):
    """
    MLP probe that predicts per-task accuracy from average loss.

    This is the neural network equivalent of the logistic scaling law baseline.
    While the logistic baseline fits: accuracy = f(avg_loss) using a 4-parameter logistic,
    this probe learns: accuracy = MLP(avg_loss) with quantile regression.

    Architecture: Small MLP -> Per-task Predictions

    This model has very high bias because:
    1. It only sees a single scalar (average loss)
    2. The MLP is intentionally small
    3. But it can learn non-linear relationships beyond logistic
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        loss_type: LossType = "mse",
        num_tasks: int = 32,
        quantiles: List[float] = None,
        inverse_perplexity: bool = False,
    ):
        """
        Args:
            hidden_dims: Tuple of hidden layer dimensions
            loss_type: "mse" or "quantile"
            num_tasks: Number of downstream tasks to predict
            quantiles: List of quantiles for quantile regression
            inverse_perplexity: If True, transform input x -> e^(-x) before MLP
        """
        super().__init__()

        self.hidden_dims = hidden_dims
        self.loss_type = loss_type
        self.num_tasks = num_tasks
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.num_quantiles = len(self.quantiles)
        self.inverse_perplexity = inverse_perplexity

        # Determine output dimension per task
        if loss_type == "quantile":
            self._output_per_task = self.num_quantiles
        else:
            self._output_per_task = 1

        # Total output: num_tasks * output_per_task
        output_dim = num_tasks * self._output_per_task

        # Build MLP: input is single scalar (avg_loss)
        layers = []
        in_dim = 1  # Single scalar input
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, avg_loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            avg_loss: (batch,) or (batch, 1) average loss values
        Returns:
            - loss_type="mse": (batch, num_tasks) per-task predictions
            - loss_type="quantile": (batch, num_tasks, num_quantiles) per-task quantile predictions
        """
        # Ensure input is (batch, 1)
        if avg_loss.dim() == 1:
            avg_loss = avg_loss.unsqueeze(-1)

        # Apply inverse perplexity transform if enabled: x -> e^(-x)
        if self.inverse_perplexity:
            avg_loss = torch.exp(-avg_loss)

        output = self.mlp(avg_loss)

        if self.loss_type == "mse":
            return output  # (batch, num_tasks)
        else:
            return output.view(-1, self.num_tasks, self.num_quantiles)  # (batch, num_tasks, num_quantiles)

    def forward_median(self, avg_loss: torch.Tensor) -> torch.Tensor:
        """Get median prediction (for quantile loss)."""
        if self.loss_type != "quantile":
            raise ValueError("forward_median only valid for quantile loss")
        output = self.forward(avg_loss)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return output[:, :, median_idx]  # (batch, num_tasks)

    def compute_loss(
        self,
        avg_loss: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            avg_loss: (batch,) or (batch, 1) average loss values
            targets: (batch, num_tasks) per-task accuracy targets
        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss metrics
        """
        pred = self.forward(avg_loss)
        return compute_loss_linear(pred, targets, self.loss_type, self.quantiles)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
