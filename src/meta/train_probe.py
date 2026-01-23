"""
Unified training script for probe models.

Supports 4 probe types:
- cnn: CNN encoder for absolute accuracy prediction
- histogram: Histogram encoder for absolute accuracy prediction
- delta: Delta histogram probe for delta accuracy prediction
- kl_delta: KL divergence probe for delta accuracy prediction

Usage:
    # CNN probe
    python -m src.meta.train_probe --probe_type cnn --output_dir results/probe_cnn

    # Histogram probe
    python -m src.meta.train_probe --probe_type histogram --num_bins 32 --output_dir results/probe_hist

    # Delta probe
    python -m src.meta.train_probe --probe_type delta --num_bins 32 --output_dir results/probe_delta

    # KL Delta probe
    python -m src.meta.train_probe --probe_type kl_delta --num_bins 32 --output_dir results/probe_kl
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.meta.probes import (
    DirectProbe,
    HistogramProbe,
    DeltaHistogramProbe,
    KLDeltaProbe,
    AverageLossProbe,
    DEFAULT_QUANTILES,
)
from src.meta.datasets import create_probe_dataloaders, create_delta_probe_dataloaders
from src.analysis import CLASSIC_TASKS


@dataclass
class ProbeTrainingConfig:
    """Unified training configuration for all probe types."""
    # Probe type
    probe_type: str = "histogram"  # "cnn", "histogram", "delta", "kl_delta"

    # Data
    data_dir: str = "results/datadecide_train"
    tasks: List[str] = None
    max_seq_len: int = 100000
    val_fraction: float = 0.15
    loss_file_pattern: str = "word_losses*.npy"
    heldout_tasks: List[str] = None

    # Histogram settings (for histogram, delta, kl_delta)
    num_bins: int = 32
    bin_min: float = 0.0
    bin_max: float = 15.0
    hidden_dims: Tuple[int, ...] = (64, 32)

    # CNN-specific settings
    channels: Tuple[int, ...] = (4, 8, 16, 32)
    kernel_size: int = 16
    stride: int = 2
    pool_size: int = 32
    hidden_dim: int = 256  # For CNN head
    head_type: str = "average"  # "average" or "linear" (CNN only)

    # Delta-specific settings (for delta, kl_delta)
    min_gap: int = 1
    max_gap: Optional[int] = None

    # Shared settings
    loss_type: str = "mse"  # "mse" or "quantile"
    quantiles: Tuple[float, ...] = None
    use_rms_norm: bool = False
    inverse_perplexity: bool = False  # If True, transform losses x -> e^(-x)

    # Training
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Output
    output_dir: str = "results/probe"
    save_every: int = 1
    log_every: int = 1
    seed: int = 42

    def __post_init__(self):
        if self.tasks is None and self.probe_type in ("cnn", "histogram"):
            # Discover all tasks from data directory (excluding avg_losses_by_step)
            self.tasks = self._discover_tasks()

    def _discover_tasks(self) -> List[str]:
        """Discover all available tasks from the data directory."""
        data_path = Path(self.data_dir)
        tasks = set()

        # Look through all model directories
        for model_dir in data_path.iterdir():
            if not model_dir.is_dir():
                continue
            # Find all .npy files (task accuracy files)
            for npy_file in model_dir.glob("*.npy"):
                task_name = npy_file.stem
                if task_name != "avg_losses_by_step":
                    tasks.add(task_name)

        if not tasks:
            print(f"Warning: No tasks found in {self.data_dir}, falling back to CLASSIC_TASKS")
            return list(CLASSIC_TASKS)

        return sorted(tasks)

    @property
    def is_delta_probe(self) -> bool:
        """Check if this is a delta-style probe."""
        return self.probe_type in ("delta", "kl_delta")


class ProbeTrainer:
    """Unified trainer for all probe types."""

    def __init__(self, config: ProbeTrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # Set seed
        torch.manual_seed(config.seed)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Create dataloaders based on probe type
        print("Creating dataloaders...")
        if config.is_delta_probe:
            print(f"  Gap range: [{config.min_gap}, {config.max_gap or 'unlimited'}]")
            if config.inverse_perplexity:
                print(f"  Inverse perplexity transform: enabled")
            self.train_loader, self.val_loader = create_delta_probe_dataloaders(
                data_dir=config.data_dir,
                tasks=config.tasks,
                batch_size=config.batch_size,
                num_workers=4,
                seed=config.seed,
                val_fraction=config.val_fraction,
                loss_file_pattern=config.loss_file_pattern,
                heldout_tasks=config.heldout_tasks,
                num_bins=config.num_bins,
                bin_min=config.bin_min,
                bin_max=config.bin_max,
                min_gap=config.min_gap,
                max_gap=config.max_gap,
                inverse_perplexity=config.inverse_perplexity,
            )
            # Store task names from dataset
            self.tasks = self.train_loader.dataset.tasks
        else:
            print(f"  Loss file pattern: {config.loss_file_pattern}")
            self.train_loader, self.val_loader = create_probe_dataloaders(
                data_dir=config.data_dir,
                tasks=config.tasks,
                max_seq_len=config.max_seq_len,
                batch_size=config.batch_size,
                num_workers=4,
                seed=config.seed,
                val_fraction=config.val_fraction,
                loss_file_pattern=config.loss_file_pattern,
                heldout_tasks=config.heldout_tasks,
                inverse_perplexity=config.inverse_perplexity,
            )
            # Store task names from dataset (handles auto-discovery when config.tasks is None)
            self.tasks = self.train_loader.dataset.tasks

        print(f"Tasks: {len(self.tasks)}")
        print(f"Train samples: {len(self.train_loader.dataset)}, Val samples: {len(self.val_loader.dataset)}")

        # Create model
        quantiles = list(config.quantiles) if config.quantiles else None
        self.model = self._create_model(config, quantiles)
        self.model = self.model.to(device)

        print(f"Model parameters: {self.model.num_parameters():,}")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Create scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.01,
        )

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "train_mae": [],
            "val_loss": [],
            "val_mae": [],
            "val_correlation": [],
        }

    def _create_model(self, config: ProbeTrainingConfig, quantiles: Optional[List[float]]):
        """Create the appropriate model based on probe type."""
        if config.probe_type == "cnn":
            print(f"Creating CNN probe"
                  f"loss_type='{config.loss_type}', use_rms_norm={config.use_rms_norm}, "
                  f"inverse_perplexity={config.inverse_perplexity}...")
            return DirectProbe(
                loss_type=config.loss_type,
                num_tasks=len(self.tasks),
                quantiles=quantiles,
                input_len=config.max_seq_len,
                channels=config.channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                hidden_dim=config.hidden_dim,
            )
        elif config.probe_type == "histogram":
            print(f"Creating histogram probe with num_bins={config.num_bins}, "
                  f"hidden_dims={config.hidden_dims}, loss_type='{config.loss_type}', "
                  f"use_rms_norm={config.use_rms_norm}...")
            return HistogramProbe(
                num_bins=config.num_bins,
                hidden_dims=config.hidden_dims,
                loss_type=config.loss_type,
                num_tasks=len(self.tasks),
                quantiles=quantiles,
                use_rms_norm=config.use_rms_norm,
            )
        elif config.probe_type == "avg_loss":
            print(f"Creating avg_loss probe with hidden_dims={config.hidden_dims}, "
                  f"loss_type='{config.loss_type}', inverse_perplexity={config.inverse_perplexity}, "
                  f"num_tasks={len(self.tasks)}...")
            return AverageLossProbe(
                hidden_dims=config.hidden_dims,
                loss_type=config.loss_type,
                num_tasks=len(self.tasks),
                quantiles=quantiles,
                inverse_perplexity=config.inverse_perplexity,
            )
        elif config.probe_type == "kl_delta":
            print(f"Creating kl_delta probe with num_bins={config.num_bins}, "
                  f"hidden_dims={config.hidden_dims}, loss_type='{config.loss_type}', "
                  f"num_tasks={len(self.tasks)}...")
            return KLDeltaProbe(
                num_bins=config.num_bins,
                hidden_dims=config.hidden_dims,
                loss_type=config.loss_type,
                num_tasks=len(self.tasks),
                quantiles=quantiles,
            )
        else:  # delta
            print(f"Creating delta probe with num_bins={config.num_bins}, "
                  f"hidden_dims={config.hidden_dims}, loss_type='{config.loss_type}', "
                  f"num_tasks={len(self.tasks)}...")
            return DeltaHistogramProbe(
                num_bins=config.num_bins,
                hidden_dims=config.hidden_dims,
                loss_type=config.loss_type,
                num_tasks=len(self.tasks),
                quantiles=quantiles,
            )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_mae = 0
        num_batches = 0
        is_kl_probe = isinstance(self.model, KLDeltaProbe)
        is_delta = self.config.is_delta_probe

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            if is_delta:
                # Delta probes predict per-task delta accuracy
                targets = batch["delta_accuracy"].to(self.device)  # (batch, num_tasks)
                if is_kl_probe:
                    hist_curr = batch["histogram_curr"].to(self.device)
                    hist_prev = batch["histogram_prev"].to(self.device)
                    loss, metrics = self.model.compute_loss(hist_curr, hist_prev, targets)
                else:
                    delta_histogram = batch["delta_histogram"].to(self.device)
                    loss, metrics = self.model.compute_loss(delta_histogram, targets)
                mae = metrics.get("mae", 0)
            else:
                if self.config.probe_type == "histogram":
                    hist = batch["histogram"].to(self.device)
                    targets = batch["task_accuracies"].to(self.device)
                    loss, metrics = self.model.compute_loss(hist, targets)
                    mae = metrics.get("mae", 0)
                elif self.config.probe_type == "avg_loss":
                    avg_loss = batch["avg_loss"].to(self.device)
                    targets = batch["task_accuracies"].to(self.device)
                    loss, metrics = self.model.compute_loss(avg_loss, targets)
                    mae = metrics.get("mae", 0)
                else:
                    losses = batch["losses"].to(self.device)
                    targets = batch["task_accuracies"].to(self.device)
                    loss, metrics = self.model.compute_loss(losses, targets)
                    mae = metrics.get("mae_avg", metrics.get("mae", 0))

            # Backward
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            self.step += 1

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        return {"loss": avg_loss, "mae": avg_mae}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        total_mae = 0
        all_preds = []
        all_targets = []
        num_batches = 0
        is_kl_probe = isinstance(self.model, KLDeltaProbe)
        is_delta = self.config.is_delta_probe

        for batch in self.val_loader:
            if is_delta:
                targets = batch["delta_accuracy"].to(self.device)  # (batch, num_tasks)
                if is_kl_probe:
                    hist_curr = batch["histogram_curr"].to(self.device)
                    hist_prev = batch["histogram_prev"].to(self.device)
                    loss, metrics = self.model.compute_loss(hist_curr, hist_prev, targets)
                    pred = self.model.forward(hist_curr, hist_prev)
                else:
                    delta_histogram = batch["delta_histogram"].to(self.device)
                    loss, metrics = self.model.compute_loss(delta_histogram, targets)
                    pred = self.model.forward(delta_histogram)

                if self.config.loss_type == "quantile":
                    median_idx = self.model.quantiles.index(0.5) if 0.5 in self.model.quantiles else len(self.model.quantiles) // 2
                    pred = pred[:, :, median_idx]  # (batch, num_tasks)
            else:
                task_targets = batch["task_accuracies"].to(self.device)

                if self.config.probe_type == "histogram":
                    # Histogram probe now predicts per-task
                    hist = batch["histogram"].to(self.device)
                    loss, metrics = self.model.compute_loss(hist, task_targets)
                    pred = self.model.forward(hist)  # (batch, num_tasks) or (batch, num_tasks, num_quantiles)
                    targets = task_targets
                elif self.config.probe_type == "avg_loss":
                    # Avg loss probe predicts per-task from scalar avg_loss
                    avg_loss_val = batch["avg_loss"].to(self.device)
                    loss, metrics = self.model.compute_loss(avg_loss_val, task_targets)
                    pred = self.model.forward(avg_loss_val)  # (batch, num_tasks) or (batch, num_tasks, num_quantiles)
                    targets = task_targets
                else:
                    losses = batch["losses"].to(self.device)
                    loss, metrics = self.model.compute_loss(losses, task_targets)
                    pred = self.model.forward(losses)  # (batch, num_tasks) or (batch, num_tasks, num_quantiles)
                    targets = task_targets

                # Extract median prediction for quantile loss type
                if self.config.loss_type == "quantile" and pred.dim() == 3:
                    median_idx = self.model.quantiles.index(0.5) if 0.5 in self.model.quantiles else len(self.model.quantiles) // 2
                    pred = pred[:, :, median_idx]  # (batch, num_tasks)

            total_loss += loss.item()
            total_mae += metrics.get("mae", metrics.get("mae_avg", 0))
            all_preds.append(pred.cpu())
            all_targets.append(targets.cpu())
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        # Compute correlation
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        pred_centered = all_preds - all_preds.mean()
        target_centered = all_targets - all_targets.mean()
        correlation = (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm() + 1e-8
        )

        return {
            "loss": avg_loss,
            "mae": avg_mae,
            "correlation": correlation.item(),
        }

    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print()

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_correlation"].append(val_metrics["correlation"])

            # Log
            if (epoch + 1) % self.config.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                # Use 6 decimal places for delta probes (smaller values), 4 for others
                fmt = ".6f" if self.config.is_delta_probe else ".4f"
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:{fmt}} | "
                    f"Train MAE: {train_metrics['mae']:{fmt}} | "
                    f"Val Loss: {val_metrics['loss']:{fmt}} | "
                    f"Val MAE: {val_metrics['mae']:{fmt}} | "
                    f"Val Corr: {val_metrics['correlation']:.4f} | "
                    f"LR: {lr:.2e}"
                )

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
                fmt = ".6f" if self.config.is_delta_probe else ".4f"
                print(f"  -> New best model (val_loss: {val_metrics['loss']:{fmt}})")

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final model
        self.save_checkpoint("final.pt")

        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        elapsed = time.time() - start_time
        fmt = ".6f" if self.config.is_delta_probe else ".4f"
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:{fmt}}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "tasks": self.tasks,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]


def main():
    parser = argparse.ArgumentParser(description="Train probe model")

    # Probe type
    parser.add_argument("--probe_type", type=str, default="histogram",
                        choices=["cnn", "histogram", "delta", "kl_delta", "avg_loss"],
                        help="Probe type: 'cnn', 'histogram', 'delta', 'kl_delta', or 'avg_loss'")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="results/datadecide_train")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--target_list", type=str, nargs="+", default=None,
                        help="Alias for --tasks")
    parser.add_argument("--max_seq_len", type=int, default=256000)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--loss_file_pattern", type=str, default="word_losses*.npy")
    parser.add_argument("--heldout_tasks", type=str, nargs="+", default=None,
                        help="Tasks to exclude from training")

    # Histogram arguments (for histogram, delta, kl_delta)
    parser.add_argument("--num_bins", type=int, default=32,
                        help="Number of histogram bins")
    parser.add_argument("--bin_min", type=float, default=0.0,
                        help="Minimum loss value for histogram binning")
    parser.add_argument("--bin_max", type=float, default=15.0,
                        help="Maximum loss value for histogram binning")
    parser.add_argument("--hidden_dims", type=str, default="64,32",
                        help="Hidden layer dimensions (comma-separated, e.g., '64,32')")

    # CNN-specific arguments
    parser.add_argument("--channels", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--kernel_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--pool_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for CNN head")
    parser.add_argument("--head_type", type=str, default="average",
                        choices=["average", "linear"],
                        help="Head type for CNN: 'average' for scalar output, 'linear' for per-task outputs")

    # Delta-specific arguments (for delta, kl_delta)
    parser.add_argument("--min_gap", type=int, default=1,
                        help="Minimum gap between step indices (1 = consecutive only)")
    parser.add_argument("--max_gap", type=int, default=None,
                        help="Maximum gap between step indices (None = unlimited)")

    # Shared arguments
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "quantile"],
                        help="Loss type: 'mse' for MSE loss, 'quantile' for pinball loss")
    parser.add_argument("--quantiles", type=float, nargs="+", default=None,
                        help="Quantiles for quantile loss (default: 0.1, 0.25, 0.5, 0.75, 0.9)")
    parser.add_argument("--use_rms_norm", action="store_true",
                        help="Apply RMS normalization to features before head")
    parser.add_argument("--inverse_perplexity", action="store_true",
                        help="Transform losses x -> e^(-x) before processing (values become 0-1 range)")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/probe")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Handle quantiles
    quantiles = None
    if args.loss_type == "quantile":
        quantiles = tuple(args.quantiles) if args.quantiles else tuple(DEFAULT_QUANTILES)

    # Handle --target_list as alias for --tasks
    tasks = args.tasks or args.target_list

    # Parse hidden_dims from comma-separated string
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    # Create config
    config = ProbeTrainingConfig(
        probe_type=args.probe_type,
        data_dir=args.data_dir,
        tasks=tasks,
        max_seq_len=args.max_seq_len,
        val_fraction=args.val_fraction,
        loss_file_pattern=args.loss_file_pattern,
        heldout_tasks=args.heldout_tasks,
        num_bins=args.num_bins,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        hidden_dims=hidden_dims,
        channels=tuple(args.channels),
        kernel_size=args.kernel_size,
        stride=args.stride,
        pool_size=args.pool_size,
        hidden_dim=args.hidden_dim,
        head_type=args.head_type,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        loss_type=args.loss_type,
        quantiles=quantiles,
        use_rms_norm=args.use_rms_norm,
        inverse_perplexity=args.inverse_perplexity,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
    )

    # Train
    trainer = ProbeTrainer(config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
