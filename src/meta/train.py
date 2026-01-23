"""
Training script for the MetaLossPredictor model.

This script trains a CNN-based soft prompt generator and a transformer
encoder from scratch. The model learns to predict future validation losses
given context losses and a query gap.

Usage:
    python -m src.meta.train \
        --data_dir ./results/datadecide_eval \
        --output_dir ./results/metaloss_predictor \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --num_epochs 10
"""

import argparse
import logging
import os
import json
from typing import Tuple, Optional, List
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

import numpy as np

from src.meta.model import (
    MetaLossPredictor, MetaLossPredictorBaseline,
)
from src.meta.datasets import (
    MetaLossDataset, MetaAccuracyDataset,
    MetaTaskAccuracyDataset, collate_fn, collate_fn_accuracy,
    MetaAccuracyDatasetAvgLoss, MetaAccuracyDatasetDeltaHist
)


def worker_init_fn(worker_id: int):
    """
    Initialize worker with unique seed based on worker_id and PyTorch's initial seed.

    This ensures different random states across workers AND across epochs,
    since torch.initial_seed() changes each epoch when shuffle=True.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    """Configuration for training the MetaLossPredictor."""
    # Reproducibility
    seed: int = 42

    # Data
    data_dir: str = "./results/datadecide_eval"
    output_dir: str = "./results/metaloss_predictor"
    predict_accuracy: bool = False  # If True, predict accuracy instead of loss
    target_list: Optional[List[str]] = None  # For accuracy: tasks to predict
    heldout_list: Optional[List[str]] = None  # For accuracy: tasks to exclude
    # Encoder input type: what to feed into the encoder
    # - "loss": raw token-level losses (default, original behavior)
    # - "instance_accuracy": instance-level binary accuracies from eval tasks
    # - "task_accuracy": task-level accuracies (strongest diagnostic - answer in input)
    input_type: str = "loss"
    instance_subdir: str = "instance_accuracies"  # Subdirectory for instance accuracy files
    min_context_steps: int = 4
    max_query_gap: int = 10
    max_encoder_tokens: int = 8192
    drop_prob: float = 0.0  # Data augmentation dropout probability
    loss_file_pattern: str = "word_losses*.npy"  # Glob pattern for loss files
    max_clip: float = 20.0  # Max clip for loss values (use 100 for word losses)
    # Inverse perplexity transform: x -> e^(-x), maps losses to (0, 1) range
    inverse_perplexity: bool = False
    bin_min: float = 0.0  # Min value for histogram binning (after transform if inverse_perplexity)
    bin_max: float = 15.0  # Max value for histogram binning (1.0 if inverse_perplexity)
    # Model
    use_baseline: bool = False  # Baseline without CNN (just transformer encoder)
    loss_type: str = "mse"  # "mse" or "quantile"
    quantiles: Optional[List[float]] = None  # For quantile regression
    # Architecture type
    architecture_type: str = "encoder"  # Only "encoder" (bidirectional with RoPE) is supported
    # Transformer config
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    max_seq_len: int = 512
    # Soft prompt generator config
    soft_prompt_type: str = "cnn"  # "cnn", "avg_loss", "delta"
    num_bins: int = 64  # For histogram/delta variants
    # CNN config (only used when soft_prompt_type="cnn")
    cnn_channels: Tuple[int, ...] = (4, 8, 16, 32)
    cnn_kernel_size: int = 16
    cnn_stride: int = 8

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.033
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 1000
    save_steps: int = 1000
    use_wandb: bool = True
    wandb_project: str = "metaloss-predictor"
    wandb_run_name: Optional[str] = None
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    num_workers: int = 4
    
    max_samples: Optional[int] = None


class MetaLossTrainer:
    """Trainer class for the MetaLossPredictor model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed for reproducibility
        set_seed(config.seed)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Initialize model
        self._init_model()
        
        # Initialize datasets
        self._init_datasets()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
    
    def _init_model(self):
        """Initialize the model."""
        if self.config.use_baseline:
            model_type = "baseline"
        else:
            model_type = "full"
        self.logger.info(f"Initializing model: type={model_type}, loss_type={self.config.loss_type}")

        # Common kwargs for transformer encoder
        encoder_kwargs = dict(
            loss_type=self.config.loss_type,
            quantiles=self.config.quantiles,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            max_seq_len=self.config.max_seq_len,
            max_gap=self.config.max_query_gap,
        )

        # Soft prompt generator kwargs
        encoder_kwargs_soft_prompt = {
            "soft_prompt_type": self.config.soft_prompt_type,
            "num_bins": self.config.num_bins,
            # Architecture type
            "architecture_type": self.config.architecture_type,
            # CNN config (only used when soft_prompt_type="cnn")
            "cnn_input_len": self.config.max_encoder_tokens,
            "cnn_channels": self.config.cnn_channels,
            "cnn_kernel_size": self.config.cnn_kernel_size,
            "cnn_stride": self.config.cnn_stride,
        }

        if self.config.use_baseline:
            # Baseline: just transformer encoder, no soft prompt generator
            self.model = MetaLossPredictorBaseline(**encoder_kwargs)
        else:
            # Full model with soft prompt generator
            model_kwargs = {
                **encoder_kwargs,
                **encoder_kwargs_soft_prompt,
            }
            self.model = MetaLossPredictor(**model_kwargs)
        
        self.model = self.model.to(self.device)
        
        # Log parameter counts
        trainable = self.model.num_trainable_parameters()
        total = self.model.num_total_parameters()
        self.logger.info(f"Trainable parameters: {trainable:,}")
        self.logger.info(f"Total parameters: {total:,}")
        
    def _init_datasets(self):
        """Initialize train and validation datasets."""
        self.logger.info("Loading datasets...")
        self.logger.info(f"Input type: {self.config.input_type}")
        if self.config.inverse_perplexity:
            self.logger.info(f"Inverse perplexity transform: enabled (bin_min={self.config.bin_min}, bin_max={self.config.bin_max})")

        if self.config.input_type == "task_accuracy":
            # Task accuracy mode: encoder input = vector of all task accuracies at current step
            self.logger.info("Using MetaTaskAccuracyDataset")
            self.train_dataset = MetaTaskAccuracyDataset(
                data_dir=self.config.data_dir,
                min_context_steps=self.config.min_context_steps,
                max_query_gap=self.config.max_query_gap,
                max_encoder_tokens=self.config.max_encoder_tokens,
                split="train",
                drop_prob=self.config.drop_prob,
                target_list=self.config.target_list,
                heldout_list=self.config.heldout_list,
            )

            # Share pre-built samples with val dataset to avoid rebuilding index
            self.val_dataset = MetaTaskAccuracyDataset(
                data_dir=self.config.data_dir,
                min_context_steps=self.config.min_context_steps,
                max_query_gap=self.config.max_query_gap,
                max_encoder_tokens=self.config.max_encoder_tokens,
                split="val",
                drop_prob=0.0,
                target_list=self.config.target_list,
                heldout_list=self.config.heldout_list,
                _shared_samples=self.train_dataset._all_samples,
            )

            collate_fn_to_use = collate_fn_accuracy
        elif self.config.predict_accuracy:
            # Select dataset class based on soft_prompt_type
            if self.config.soft_prompt_type == "avg_loss":
                self.logger.info("Using MetaAccuracyDatasetAvgLoss")
                DatasetClass = MetaAccuracyDatasetAvgLoss
                extra_kwargs = {}
            elif self.config.soft_prompt_type == "delta":
                self.logger.info("Using MetaAccuracyDatasetDeltaHist")
                DatasetClass = MetaAccuracyDatasetDeltaHist
                extra_kwargs = {
                    "num_bins": self.config.num_bins,
                    "inverse_perplexity": self.config.inverse_perplexity,
                }
            else:
                # Default: cnn uses MetaAccuracyDataset with raw token losses
                self.logger.info("Using MetaAccuracyDataset")
                DatasetClass = MetaAccuracyDataset
                # Pass inverse_perplexity params for raw loss processing
                extra_kwargs = {
                    "inverse_perplexity": self.config.inverse_perplexity,
                    "bin_min": self.config.bin_min,
                    "bin_max": self.config.bin_max,
                }

            self.train_dataset = DatasetClass(
                data_dir=self.config.data_dir,
                min_context_steps=self.config.min_context_steps,
                max_query_gap=self.config.max_query_gap,
                max_encoder_tokens=self.config.max_encoder_tokens,
                split="train",
                drop_prob=self.config.drop_prob,
                target_list=self.config.target_list,
                heldout_list=self.config.heldout_list,
                **extra_kwargs,
            )

            # Share pre-built samples with val dataset to avoid rebuilding index
            # Also share caches for MetaAccuracyDatasetAvgLoss and MetaAccuracyDatasetDeltaHist
            shared_cache_kwargs = {}
            if hasattr(self.train_dataset, '_avg_losses_cache'):
                shared_cache_kwargs['_shared_avg_losses_cache'] = self.train_dataset._avg_losses_cache
            if hasattr(self.train_dataset, '_histograms_cache'):
                shared_cache_kwargs['_shared_histograms_cache'] = self.train_dataset._histograms_cache

            self.val_dataset = DatasetClass(
                data_dir=self.config.data_dir,
                min_context_steps=self.config.min_context_steps,
                max_query_gap=self.config.max_query_gap,
                max_encoder_tokens=self.config.max_encoder_tokens,
                split="val",
                drop_prob=0.0,  # No augmentation for validation
                target_list=self.config.target_list,
                heldout_list=self.config.heldout_list,
                _shared_samples=self.train_dataset._all_samples,
                **shared_cache_kwargs,
                **extra_kwargs,
            )

            collate_fn_to_use = collate_fn_accuracy
        else:
            self.logger.info("Using MetaLossDataset")
            self.logger.info(f"Loss file pattern: {self.config.loss_file_pattern}")
            self.train_dataset = MetaLossDataset(
                data_dir=self.config.data_dir,
                min_context_steps=self.config.min_context_steps,
                max_query_gap=self.config.max_query_gap,
                max_encoder_tokens=self.config.max_encoder_tokens,
                split="train",
                drop_prob=self.config.drop_prob,
                loss_file_pattern=self.config.loss_file_pattern,
                max_clip=self.config.max_clip,
            )

            # Share pre-built samples with val dataset to avoid rebuilding index
            self.val_dataset = MetaLossDataset(
                data_dir=self.config.data_dir,
                min_context_steps=self.config.min_context_steps,
                max_query_gap=self.config.max_query_gap,
                max_encoder_tokens=self.config.max_encoder_tokens,
                split="val",
                drop_prob=0.0,  # No augmentation for validation
                loss_file_pattern=self.config.loss_file_pattern,
                max_clip=self.config.max_clip,
                _shared_samples=self.train_dataset._all_samples,
            )

            collate_fn_to_use = collate_fn
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn_to_use,
            worker_init_fn=worker_init_fn,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_to_use,
            worker_init_fn=worker_init_fn,
        )
        
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Warmup + Cosine decay scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.01,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )
        
        self.logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        global_step = 0
        best_val_loss = float("inf")
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler() if self.config.fp16 else None
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics = {}
            num_batches = 0
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )
            
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                encoder_input = batch["encoder_input"].to(self.device)
                # Support both loss and accuracy field names
                context_key = "context_losses" if "context_losses" in batch else "context_values"
                target_key = "target_loss" if "target_loss" in batch else "target_value"
                context_losses = batch[context_key].to(self.device)
                context_gaps = batch["context_gaps"].to(self.device)
                context_mask = batch["context_mask"].to(self.device)
                target_loss = batch[target_key].to(self.device)
                # Get encoder mask if present (for variable-length soft prompts)
                encoder_mask = batch.get("encoder_mask")
                if encoder_mask is not None:
                    encoder_mask = encoder_mask.to(self.device)

                # Forward pass with mixed precision
                if self.config.fp16:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        loss, metrics = self.model.compute_loss(
                            encoder_input=encoder_input,
                            context_losses=context_losses,
                            context_gaps=context_gaps,
                            context_mask=context_mask,
                            encoder_mask=encoder_mask,
                            target_loss=target_loss,
                        )
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss, metrics = self.model.compute_loss(
                        encoder_input=encoder_input,
                        context_losses=context_losses,
                        context_gaps=context_gaps,
                        context_mask=context_mask,
                        encoder_mask=encoder_mask,
                        target_loss=target_loss,
                    )
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate metrics
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                num_batches += 1
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "mae": f"{avg_metrics.get('mae', avg_metrics.get('median_mae', 0)):.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        })
                        
                        if self.config.use_wandb:
                            log_dict = {
                                "train/loss": avg_loss,
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            }
                            for k, v in avg_metrics.items():
                                log_dict[f"train/{k}"] = v
                            wandb.log(log_dict)
                    
                    # Evaluation
                    if global_step % self.config.eval_steps == 0:
                        val_loss, val_metrics = self.evaluate()
                        
                        self.logger.info(
                            f"Step {global_step}: val_loss={val_loss:.4f}, "
                            f"val_mae={val_metrics.get('mae', val_metrics.get('median_mae', 0)):.4f}"
                        )
                        
                        if self.config.use_wandb:
                            log_dict = {
                                "val/loss": val_loss,
                                "global_step": global_step,
                            }
                            for k, v in val_metrics.items():
                                log_dict[f"val/{k}"] = v
                            wandb.log(log_dict)
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint("best_model.pt", global_step, val_loss)
                        
                        self.model.train()
                    
                    # Regular checkpointing
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint_{global_step}.pt", global_step)
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint("final_model.pt", global_step)
        
        if self.config.use_wandb:
            wandb.finish()
            
        self.logger.info("Training completed!")
        
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, dict]:
        """Evaluate the model on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_metrics = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            encoder_input = batch["encoder_input"].to(self.device)
            # Support both loss and accuracy field names
            context_key = "context_losses" if "context_losses" in batch else "context_values"
            target_key = "target_loss" if "target_loss" in batch else "target_value"
            context_losses = batch[context_key].to(self.device)
            context_gaps = batch["context_gaps"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            target_loss = batch[target_key].to(self.device)
            # Get encoder mask if present (for variable-length soft prompts)
            encoder_mask = batch.get("encoder_mask")
            if encoder_mask is not None:
                encoder_mask = encoder_mask.to(self.device)

            if self.config.fp16:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    loss, metrics = self.model.compute_loss(
                        encoder_input=encoder_input,
                        context_losses=context_losses,
                        context_gaps=context_gaps,
                        context_mask=context_mask,
                        encoder_mask=encoder_mask,
                        target_loss=target_loss,
                    )
            else:
                loss, metrics = self.model.compute_loss(
                    encoder_input=encoder_input,
                    context_losses=context_losses,
                    context_gaps=context_gaps,
                    context_mask=context_mask,
                    encoder_mask=encoder_mask,
                    target_loss=target_loss,
                )

            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, filename: str, global_step: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": global_step,
            "config": asdict(self.config),
        }
        
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get("global_step", 0)


def main():
    parser = argparse.ArgumentParser(description="Train MetaLossPredictor model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./results/datadecide_eval",
                        help="Directory containing preprocessed loss data")
    parser.add_argument("--output_dir", type=str, default="./results/metaloss_predictor",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--predict_accuracy", action="store_true",
                        help="Predict accuracy instead of loss")
    parser.add_argument("--target_list", type=str, nargs="+", default=None,
                        help="For accuracy prediction: only predict these tasks (default: all)")
    parser.add_argument("--heldout_list", type=str, nargs="+", default=None,
                        help="For accuracy prediction: tasks to exclude (default: avg_losses_by_step)")
    parser.add_argument("--input_type", type=str, default="loss",
                        choices=["loss", "instance_accuracy", "task_accuracy"],
                        help="Encoder input type: loss (raw token losses), instance_accuracy (binary per-instance), task_accuracy (aggregate per-task)")
    parser.add_argument("--instance_subdir", type=str, default="instance_accuracies",
                        help="Subdirectory for instance accuracy files")
    parser.add_argument("--min_context_steps", type=int, default=1,
                        help="Minimum context steps")
    parser.add_argument("--max_query_gap", type=int, default=10,
                        help="Maximum query gap (how far ahead to predict)")
    parser.add_argument("--max_encoder_tokens", type=int, default=128000,
                        help="Maximum tokens for encoder input (only for loss prediction)")
    parser.add_argument("--drop_prob", type=float, default=0.4,
                        help="Dropout probability for data augmentation")
    parser.add_argument("--loss_file_pattern", type=str, default="word_losses*.npy")
    parser.add_argument("--max_clip", type=float, default=20.0,
                        help="Max clip value for losses (use 100 for word-level losses which are sums)")
    parser.add_argument("--inverse_perplexity", action="store_true",
                        help="Transform losses x -> e^(-x) before processing (values become 0-1 range)")
    parser.add_argument("--bin_min", type=float, default=0.0,
                        help="Min value for histogram binning (default 0.0)")
    parser.add_argument("--bin_max", type=float, default=15.0,
                        help="Max value for histogram binning (default 15.0, use 1.0 with inverse_perplexity)")
    # Model arguments
    parser.add_argument("--use_baseline", action="store_true",
                        help="Use baseline model without CNN (just transformer encoder)")
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "quantile"],
                        help="Loss type: mse or quantile")
    parser.add_argument("--quantiles", type=float, nargs="+", default=None,
                        help="Quantiles for quantile regression (e.g., 0.1 0.5 0.9)")
    # Architecture type (only encoder is supported)
    parser.add_argument("--architecture_type", type=str, default="encoder",
                        choices=["encoder"],
                        help="Architecture type: encoder (bidirectional with RoPE)")
    # Transformer arguments
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Transformer hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=2048,
                        help="Transformer feedforward dimension")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    # Soft prompt generator arguments
    parser.add_argument("--soft_prompt_type", type=str, default="cnn",
                        choices=["cnn", "avg_loss", "delta"],
                        help="Type of soft prompt generator (default: cnn)")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="Number of histogram bins (for delta)")
    # CNN arguments (only used when soft_prompt_type=cnn)
    parser.add_argument("--cnn_channels", type=str, default="8,16,32,64",
                        help="Comma-separated list of CNN channel sizes")
    parser.add_argument("--cnn_kernel_size", type=int, default=64,
                        help="CNN kernel size")
    parser.add_argument("--cnn_stride", type=int, default=16,
                        help="CNN stride")
    # Signal propagation arguments
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.033,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=2000,
                        help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=2000,
                        help="Checkpoint save frequency")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="metaloss-predictor",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16",
                        help="Disable FP16 mixed precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num workers")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Debug arguments
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples (for debugging)")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        predict_accuracy=args.predict_accuracy,
        target_list=args.target_list,
        heldout_list=args.heldout_list,
        input_type=args.input_type,
        instance_subdir=args.instance_subdir,
        min_context_steps=args.min_context_steps,
        max_query_gap=args.max_query_gap,
        max_encoder_tokens=args.max_encoder_tokens,
        drop_prob=args.drop_prob,
        loss_file_pattern=args.loss_file_pattern,
        max_clip=args.max_clip,
        inverse_perplexity=args.inverse_perplexity,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        use_baseline=args.use_baseline,
        loss_type=args.loss_type,
        quantiles=args.quantiles,
        architecture_type=args.architecture_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        soft_prompt_type=args.soft_prompt_type,
        num_bins=args.num_bins,
        cnn_channels=tuple(int(x) for x in args.cnn_channels.split(",")),
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_stride=args.cnn_stride,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        device=args.device,
        fp16=args.fp16,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    
    # Validate loss type
    if config.loss_type not in ("mse", "quantile"):
        parser.error(f"Only 'mse' or 'quantile' loss supported, got '{config.loss_type}'")
    
    # Train
    trainer = MetaLossTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    trainer.train()


if __name__ == "__main__":
    main()
