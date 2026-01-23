"""
Meta-model architecture for predicting future validation losses.

The architecture consists of:
1. A soft prompt generator (CNN, avg_loss, or delta) that encodes input into soft prompts
2. A transformer encoder that processes context losses with gap information
3. A regression output head that predicts the target loss

Supports multiple loss functions:
- MSE: Mean squared error
- Quantile: Quantile regression with pinball loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Literal

from src.meta.utils import pinball_loss


LossType = Literal["mse", "quantile"]
SoftPromptType = Literal["cnn", "avg_loss", "delta"]


class AverageLossSoftPromptGenerator(nn.Module):
    """
    Generates a single soft prompt from average loss values.

    Projects each loss value and mean-pools to produce one soft prompt.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.loss_proj = nn.Linear(1, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.loss_proj.weight)
        nn.init.zeros_(self.loss_proj.bias)

    def forward(self, avg_losses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            avg_losses: (batch, num_losses) average loss values

        Returns:
            (batch, 1, hidden_dim) single soft prompt
        """
        # (batch, num_losses) -> (batch, num_losses, 1) -> (batch, num_losses, hidden_dim)
        x = self.loss_proj(avg_losses.unsqueeze(-1))
        # Mean pool over num_losses dimension -> (batch, hidden_dim)
        x = x.mean(dim=1)
        # Return as (batch, 1, hidden_dim)
        return x.unsqueeze(1)


class DeltaProbeSoftPromptGenerator(nn.Module):
    """
    Generates a single soft prompt from histogram delta (query - context_end).

    This is used for the delta probe soft prompt type where:
    - Input: delta_histogram = histogram[query_step] - histogram[context_end]
    - Output: single (batch, 1, hidden_dim) soft prompt token

    The soft prompt is placed as the last token before prediction, allowing
    the model to use the delta histogram information to adjust its prediction.
    """

    def __init__(
        self,
        num_bins: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(num_bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, delta_histogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_histogram: (batch, num_bins) histogram delta

        Returns:
            (batch, 1, hidden_dim) single soft prompt token
        """
        # (batch, num_bins) -> (batch, hidden_dim) -> (batch, 1, hidden_dim)
        x = self.mlp(delta_histogram)
        return x.unsqueeze(1)


class CNNSoftPromptGenerator(nn.Module):
    """
    Uses 1D CNN to encode raw loss values into a single soft prompt.
    - Uses strided convolutions for downsampling
    - Flatten and project to hidden_dim
    """

    def __init__(
        self,
        input_len: int = 100000,
        hidden_dim: int = 512,
        channels: Tuple[int, ...] = (4, 8, 16, 32),
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.channels = channels
        self.stride = stride

        # Standard CNN for large inputs
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # Compute output size after conv layers
        # Conv1d output: floor((L + 2*padding - kernel_size) / stride) + 1
        padding = kernel_size // 2
        conv_out_len = input_len
        for _ in channels:
            conv_out_len = (conv_out_len + 2 * padding - kernel_size) // stride + 1

        self.conv_out_len = conv_out_len
        self.flat_dim = channels[-1] * conv_out_len

        # Project flattened features to hidden_dim (single soft prompt)
        self.proj = nn.Linear(self.flat_dim, hidden_dim)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            losses: (batch, seq_len) raw loss values
        Returns:
            (batch, 1, hidden_dim) single soft prompt
        """
        # (batch, seq_len) -> (batch, 1, seq_len)
        x = losses.unsqueeze(1)

        x = self.conv(x)  # (batch, channels[-1], conv_out_len)

        # Flatten and project: (batch, channels[-1] * conv_out_len) -> (batch, hidden_dim)
        x = x.flatten(1)
        x = self.proj(x)

        # Return as (batch, 1, hidden_dim) for consistency with soft prompt interface
        return x.unsqueeze(1)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer models.

    Applies rotation to query and key vectors based on their position,
    enabling relative position encoding through rotation operations.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for max_seq_len positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)

        # Compute cos and sin
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos_cache = emb.cos()
        sin_cache = emb.sin()

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos and sin for the given sequence length.

        Args:
            x: Input tensor (used only for device/dtype)
            seq_len: Current sequence length

        Returns:
            (cos, sin) each of shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cache[:seq_len].to(x.dtype),
            self.sin_cache[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key tensors.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine tensor of shape (seq_len, head_dim)
        sin: Sine tensor of shape (seq_len, head_dim)

    Returns:
        Rotated (q, k) tensors
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embedding (RoPE).

    Uses RoPE instead of absolute positional embeddings for better
    length generalization and relative position awareness.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Rotary position embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attn_mask: Optional attention mask (seq_len, seq_len) or (batch, seq_len, seq_len)
            key_padding_mask: Optional padding mask (batch, seq_len), True = ignore

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embedding
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_len) with -inf for padding
            # Expand to (batch, 1, 1, seq_len) for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights + key_padding_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(output)


class RoPEEncoderLayer(nn.Module):
    """
    Transformer encoder layer with RoPE attention.

    Pre-norm architecture with GELU activation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.self_attn = RoPEMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional attention mask
            key_padding_mask: Optional padding mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


class TransformerEncoderBidirectional(nn.Module):
    """
    Bidirectional transformer encoder for processing context losses with gaps.

    Uses BOS/CLS token + Rotary Position Embeddings (RoPE) with bidirectional attention.
    All positions can attend to all other positions.
    Prediction is decoded from the BOS/CLS token representation.

    Input sequence: [BOS, soft_prompt, context]
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projections for losses and gaps (each to hidden_dim/2, then concatenate)
        self.loss_projection = nn.Linear(1, hidden_dim // 2)
        self.gap_projection = nn.Linear(1, hidden_dim // 2)

        # BOS/CLS token embedding
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Bidirectional transformer encoder with RoPE
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=0.0,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize BOS embedding
        nn.init.normal_(self.bos_embedding, std=0.02)

    def forward(
        self,
        losses: Optional[torch.Tensor],
        gaps: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        soft_prompts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            losses: (batch, seq_len) loss values, or None for soft-prompt-only mode
            gaps: (batch, seq_len) gap values, or None for soft-prompt-only mode
            attention_mask: (batch, seq_len) 1 for valid, 0 for padding
            soft_prompts: Optional (batch, 1, hidden_dim) single soft prompt from generator

        Returns:
            (batch, hidden_dim) BOS/CLS token representation for prediction
        """
        # Check if we have context
        has_context = losses is not None and losses.numel() > 0

        if has_context:
            # Normal path with context
            batch_size, seq_len = losses.shape
            device = losses.device

            # Project losses and gaps, then concatenate
            loss_proj = self.loss_projection(losses.unsqueeze(-1))  # (batch, seq_len, hidden_dim/2)
            gap_proj = self.gap_projection(gaps.float().unsqueeze(-1))  # (batch, seq_len, hidden_dim/2)
            context = torch.cat([loss_proj, gap_proj], dim=-1)  # (batch, seq_len, hidden_dim)

            # Expand BOS embedding for batch
            bos = self.bos_embedding.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)

            if soft_prompts is not None:
                # [BOS | soft_prompt | context]
                x = torch.cat([bos, soft_prompts, context], dim=1)

                # Build attention mask
                bos_mask = torch.ones(batch_size, 1, device=device)
                prompt_mask = torch.ones(batch_size, soft_prompts.size(1), device=device)
                if attention_mask is not None:
                    full_mask = torch.cat([bos_mask, prompt_mask, attention_mask], dim=1)
                else:
                    context_mask = torch.ones(batch_size, seq_len, device=device)
                    full_mask = torch.cat([bos_mask, prompt_mask, context_mask], dim=1)
            else:
                # No soft prompts: [BOS | context]
                x = torch.cat([bos, context], dim=1)
                if attention_mask is not None:
                    bos_mask = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                    full_mask = torch.cat([bos_mask, attention_mask], dim=1)
                else:
                    full_mask = None
        else:
            # No context path: [BOS | soft_prompts]
            if soft_prompts is None:
                raise ValueError("Must provide soft_prompts when no context")

            batch_size = soft_prompts.size(0)
            device = soft_prompts.device

            # Expand BOS embedding for batch
            bos = self.bos_embedding.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)

            # Build sequence: just BOS + soft_prompts
            x = torch.cat([bos, soft_prompts], dim=1)

            # Build attention mask: BOS (always valid), soft prompts (always valid)
            bos_mask = torch.ones(batch_size, 1, device=device)
            prompt_mask = torch.ones(batch_size, soft_prompts.size(1), device=device)
            full_mask = torch.cat([bos_mask, prompt_mask], dim=1)

        # Convert attention mask to key_padding_mask format (float with -inf for padding)
        if full_mask is not None:
            key_padding_mask = torch.zeros_like(full_mask, dtype=x.dtype)
            key_padding_mask = key_padding_mask.masked_fill(full_mask == 0, float('-inf'))
        else:
            key_padding_mask = None

        # Apply transformer layers (bidirectional - no causal mask)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Return BOS/CLS token representation (position 0)
        return x[:, 0, :]  # (batch, hidden_dim)


class RegressionHead(nn.Module):
    """Regression head for predicting continuous loss values."""

    def __init__(self, hidden_dim: int, num_outputs: int = 1):
        super().__init__()
        self.num_outputs = num_outputs
        self.head = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_dim) pooled representation
        Returns:
            (batch,) if num_outputs=1, else (batch, num_outputs)
        """
        out = self.head(x)
        if self.num_outputs == 1:
            return out.squeeze(-1)
        return out


class MetaLossPredictor(nn.Module):
    """
    Meta-model for predicting future validation losses.

    Given:
    - Soft prompt input (raw losses, average losses, or histograms depending on soft_prompt_type)
    - Context losses with gaps
    - A query gap (how far ahead to predict)

    Predicts: A single target loss value

    Soft prompt types:
    - cnn: Raw token losses processed by CNN (original architecture)
    - avg_loss: Average loss per step projected to embeddings
    - delta: Delta histogram (query - context_end) processed by MLP

    Uses bidirectional transformer encoder with RoPE, prediction from BOS/CLS token.

    Loss types:
    - mse: Mean squared error
    - quantile: Quantile regression with pinball loss
    """

    def __init__(
        self,
        # Loss type
        loss_type: LossType = "mse",
        quantiles: Optional[List[float]] = None,  # For quantile regression
        # Transformer config
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_len: int = 512,
        # Soft prompt generator config
        soft_prompt_type: SoftPromptType = "cnn",
        num_bins: int = 32,  # For histogram variants
        # CNN-specific config (only used when soft_prompt_type="cnn")
        cnn_input_len: int = 100000,
        cnn_channels: Tuple[int, ...] = (4, 8, 16, 32),
        cnn_kernel_size: int = 16,
        cnn_stride: int = 8,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.hidden_dim = hidden_dim
        self.soft_prompt_type = soft_prompt_type

        # Validate loss type
        if loss_type not in ("mse", "quantile"):
            raise ValueError(f"Only 'mse' or 'quantile' loss supported, got '{loss_type}'")

        # Setup quantiles for quantile regression
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.register_buffer('quantiles', torch.tensor(quantiles))
        self.num_quantiles = len(quantiles)

        # Initialize soft prompt generator based on type
        # All generators output (batch, 1, hidden_dim)
        if soft_prompt_type == "cnn":
            self.soft_prompt_generator = CNNSoftPromptGenerator(
                input_len=cnn_input_len,
                hidden_dim=hidden_dim,
                channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                stride=cnn_stride,
            )
        elif soft_prompt_type == "avg_loss":
            self.soft_prompt_generator = AverageLossSoftPromptGenerator(
                hidden_dim=hidden_dim,
            )
        elif soft_prompt_type == "delta":
            self.soft_prompt_generator = DeltaProbeSoftPromptGenerator(
                num_bins=num_bins,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown soft_prompt_type: {soft_prompt_type}")

        # Bidirectional encoder with RoPE, decodes from BOS/CLS token
        self.transformer = TransformerEncoderBidirectional(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
        )

        # Output head (regression only)
        num_outputs = self.num_quantiles if loss_type == "quantile" else 1
        self.head = RegressionHead(hidden_dim, num_outputs=num_outputs)

    def forward(
        self,
        encoder_input: torch.Tensor,
        context_losses: Optional[torch.Tensor] = None,
        context_gaps: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: Input for soft prompt generator. Format depends on soft_prompt_type:
                - cnn: (batch, seq_len) raw token losses
                - avg_loss: (batch, num_losses) average loss values
                - delta: (batch, num_bins) delta histogram
            context_losses: (batch, context_len) observed losses
            context_gaps: (batch, context_len) forward gaps
            context_mask: Optional (batch, context_len)
            encoder_mask: Unused (kept for API compatibility)

        Returns:
            (batch,) predicted loss values, or (batch, num_quantiles) for quantile loss
        """

        # Generate single soft prompt: (batch, 1, hidden_dim)
        soft_prompts = self.soft_prompt_generator(encoder_input)

        # Bidirectional encoder: returns pooled BOS/CLS representation directly
        pooled = self.transformer(
            context_losses,
            context_gaps,
            attention_mask=context_mask,
            soft_prompts=soft_prompts,
        )

        # Output prediction
        return self.head(pooled)

    def compute_loss(
        self,
        encoder_input: torch.Tensor,
        context_losses: Optional[torch.Tensor],
        context_gaps: Optional[torch.Tensor],
        target_loss: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for training.

        Args:
            ... (same as forward)
            target_loss: (batch,) ground truth normalized loss values
            encoder_mask: Optional (batch, num_soft_prompts) mask for variable-length soft prompts
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics
        """
        output = self.forward(
            encoder_input=encoder_input,
            context_losses=context_losses,
            context_gaps=context_gaps,
            context_mask=context_mask,
            encoder_mask=encoder_mask,
        )

        if self.loss_type == "mse":
            main_loss = F.mse_loss(output, target_loss)
            with torch.no_grad():
                mae = F.l1_loss(output, target_loss).item()
                rmse = torch.sqrt(main_loss).item()
            metrics = {
                "main_loss": main_loss.item(),
                "mse": main_loss.item(),
                "mae": mae,
                "rmse": rmse,
            }

        elif self.loss_type == "quantile":
            main_loss = pinball_loss(output, target_loss, self.quantiles)
            with torch.no_grad():
                # Use median (closest to 0.5) for point estimate metrics
                median_idx = (self.quantiles - 0.5).abs().argmin()
                median_pred = output[:, median_idx]
                mae = F.l1_loss(median_pred, target_loss).item()
                mse = F.mse_loss(median_pred, target_loss).item()
                rmse = mse ** 0.5
            metrics = {
                "main_loss": main_loss.item(),
                "pinball_loss": main_loss.item(),
                "median_mae": mae,
                "median_mse": mse,
                "median_rmse": rmse,
            }
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = main_loss
        metrics["loss"] = loss.item()

        return loss, metrics
    
    @torch.no_grad()
    def predict(
        self,
        encoder_input: torch.Tensor,
        context_losses: Optional[torch.Tensor] = None,
        context_gaps: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_distribution: bool = False,
    ) -> torch.Tensor:
        """
        Make predictions.

        Args:
            ... (same as forward)
            return_distribution: If True, return full distribution (quantiles)

        Returns:
            For mse: (batch,) predicted values
            For quantile: (batch,) median or (batch, num_quantiles) if return_distribution
        """
        output = self.forward(
            encoder_input=encoder_input,
            context_losses=context_losses,
            context_gaps=context_gaps,
            context_mask=context_mask,
        )

        if self.loss_type == "quantile":
            if return_distribution:
                return output  # (batch, num_quantiles)
            # Return median prediction
            median_idx = (self.quantiles - 0.5).abs().argmin()
            return output[:, median_idx]
        return output

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MetaLossPredictorBaseline(nn.Module):
    """
    Baseline model without CNN - just transformer encoder on context losses.

    This serves as a baseline to measure the value added by the CNN soft prompts.
    The model only sees the context losses and gaps, without the raw token-level losses.
    """

    def __init__(
        self,
        # Loss type
        loss_type: LossType = "mse",
        quantiles: Optional[List[float]] = None,
        # Transformer config
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.hidden_dim = hidden_dim

        # Validate loss type
        if loss_type not in ("mse", "quantile"):
            raise ValueError(f"Only 'mse' or 'quantile' loss supported, got '{loss_type}'")

        # Setup quantiles for quantile regression
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.register_buffer('quantiles', torch.tensor(quantiles))
        self.num_quantiles = len(quantiles)

        # Bidirectional encoder (no soft prompts)
        self.encoder = TransformerEncoderBidirectional(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
        )

        # Output head (regression only)
        num_outputs = self.num_quantiles if loss_type == "quantile" else 1
        self.head = RegressionHead(hidden_dim, num_outputs=num_outputs)

    def forward(
        self,
        encoder_input: torch.Tensor,  # Ignored in baseline
        context_losses: torch.Tensor,
        context_gaps: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: (batch, seq_len) - IGNORED in baseline
            context_losses: (batch, context_len) observed losses
            context_gaps: (batch, context_len) forward gaps (gap[i] = distance from loss[i] to loss[i+1] or target)
            context_mask: Optional (batch, context_len)
        Returns:
            (batch,) predicted loss values, or (batch, num_quantiles) for quantile loss
        """
        # Encode context WITHOUT soft prompts (bidirectional)
        # Returns pooled BOS/CLS representation directly
        pooled = self.encoder(
            context_losses,
            context_gaps,
            attention_mask=context_mask,
            soft_prompts=None,  # No soft prompts in baseline
        )

        # Output prediction
        return self.head(pooled)

    def compute_loss(
        self,
        encoder_input: torch.Tensor,
        context_losses: torch.Tensor,
        context_gaps: torch.Tensor,
        target_loss: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for training (same signature as MetaLossPredictor).
        """
        output = self.forward(
            encoder_input=encoder_input,
            context_losses=context_losses,
            context_gaps=context_gaps,
            context_mask=context_mask,
        )

        if self.loss_type == "mse":
            loss = F.mse_loss(output, target_loss)
            with torch.no_grad():
                mae = F.l1_loss(output, target_loss).item()
                rmse = torch.sqrt(loss).item()
            metrics = {
                "loss": loss.item(),
                "mse": loss.item(),
                "mae": mae,
                "rmse": rmse,
            }

        elif self.loss_type == "quantile":
            loss = pinball_loss(output, target_loss, self.quantiles)
            with torch.no_grad():
                median_idx = (self.quantiles - 0.5).abs().argmin()
                median_pred = output[:, median_idx]
                mae = F.l1_loss(median_pred, target_loss).item()
                mse = F.mse_loss(median_pred, target_loss).item()
                rmse = mse ** 0.5
            metrics = {
                "loss": loss.item(),
                "pinball_loss": loss.item(),
                "median_mae": mae,
                "median_mse": mse,
                "median_rmse": rmse,
            }
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss, metrics

    @torch.no_grad()
    def predict(
        self,
        encoder_input: torch.Tensor,
        context_losses: torch.Tensor,
        context_gaps: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        return_distribution: bool = False,
    ) -> torch.Tensor:
        """
        Make predictions (same signature as MetaLossPredictor).
        """
        output = self.forward(
            encoder_input=encoder_input,
            context_losses=context_losses,
            context_gaps=context_gaps,
            context_mask=context_mask,
        )

        if self.loss_type == "quantile":
            if return_distribution:
                return output
            median_idx = (self.quantiles - 0.5).abs().argmin()
            return output[:, median_idx]
        return output
    
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
