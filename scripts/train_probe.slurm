#!/bin/bash
#SBATCH --job-name=probe_train
#SBATCH --output=slurm/%A_%x.out
#SBATCH --error=slurm/%A_%x.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --account=torch_pr_287_general

# =============================================================================
# Train Probe Model (Unified)
#
# Supports all probe types: cnn, histogram, delta, kl_delta
#
# Usage:
#   sbatch train_probe.slurm [OPTIONS]
#
# Options:
#   --probe_type TYPE     Probe type: 'cnn', 'histogram', 'delta', 'kl_delta' (default: histogram)
#   --loss_type TYPE      Loss type: 'mse' or 'quantile' (default: mse)
#   --batch_size N        Batch size (default: 128)
#   --epochs N            Number of epochs (default: 100)
#   --lr RATE             Learning rate (default: 1e-3)
#   --seed N              Random seed (default: 42)
#   --output_dir DIR      Output directory (default: auto-generated)
#
# CNN-specific:
#   --max_seq_len N       Max sequence length (default: 100000)
#
# Histogram-specific:
#   --num_bins N          Number of histogram bins (default: 32)
#   --hidden_dims DIMS    Hidden layer dimensions (default: 64,32)
#   --use_rms_norm        Apply RMS normalization to features
#
# Delta-specific:
#   --min_gap N           Minimum gap between step indices (default: 1)
#   --max_gap N           Maximum gap (default: unlimited)
#
# Examples:
#   sbatch train_probe.slurm --probe_type histogram --num_bins 32
#   sbatch train_probe.slurm --probe_type delta --num_bins 32 --max_gap 10
#   sbatch train_probe.slurm --probe_type kl_delta --loss_type quantile
# =============================================================================

# Defaults
PROBE_TYPE="histogram"
LOSS_TYPE="mse"
BATCH_SIZE="128"
NUM_EPOCHS="100"
LEARNING_RATE="1e-3"
SEED="42"
OUTPUT_DIR=""
DATA_DIR="./results/datadecide_train"
LOSS_FILE_PATTERN="word_losses*.npy"

# CNN-specific defaults
MAX_SEQ_LEN="256000"

# Histogram-specific defaults
NUM_BINS="32"
HIDDEN_DIMS="64,32"
USE_RMS_NORM=false

# Delta-specific defaults
MIN_GAP="1"
MAX_GAP=""

# Task filtering
TARGET_LIST=""

EXTRA_ARGS=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --probe_type)
            PROBE_TYPE="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --loss_file_pattern)
            LOSS_FILE_PATTERN="$2"
            shift 2
            ;;
        --max_seq_len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --num_bins)
            NUM_BINS="$2"
            shift 2
            ;;
        --hidden_dims)
            HIDDEN_DIMS="$2"
            shift 2
            ;;
        --use_rms_norm)
            USE_RMS_NORM=true
            shift
            ;;
        --min_gap)
            MIN_GAP="$2"
            shift 2
            ;;
        --max_gap)
            MAX_GAP="$2"
            shift 2
            ;;
        --target_list)
            TARGET_LIST="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Build run name
RUN_NAME="${PROBE_TYPE}_probe_${LOSS_TYPE}_lr${LEARNING_RATE}"

# Auto-generate output directory if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./results/${RUN_NAME}_$(date +%Y%m%d-%H%M%S)"
fi

echo "=============================================="
echo "Training Probe Model"
echo "=============================================="
echo "PROBE_TYPE: $PROBE_TYPE"
echo "LOSS_TYPE: $LOSS_TYPE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "SEED: $SEED"
echo "DATA_DIR: $DATA_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "LOSS_FILE_PATTERN: $LOSS_FILE_PATTERN"
if [[ "$PROBE_TYPE" == "histogram" || "$PROBE_TYPE" == "delta" || "$PROBE_TYPE" == "kl_delta" ]]; then
    echo "NUM_BINS: $NUM_BINS"
    echo "HIDDEN_DIMS: $HIDDEN_DIMS"
fi
if [[ "$PROBE_TYPE" == "cnn" || "$PROBE_TYPE" == "histogram" ]]; then
    echo "USE_RMS_NORM: $USE_RMS_NORM"
fi
if [[ "$PROBE_TYPE" == "delta" || "$PROBE_TYPE" == "kl_delta" ]]; then
    echo "MIN_GAP: $MIN_GAP"
    echo "MAX_GAP: ${MAX_GAP:-unlimited}"
fi
if [[ -n "$TARGET_LIST" ]]; then
    echo "TARGET_LIST: $TARGET_LIST"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "EXTRA_ARGS: $EXTRA_ARGS"
fi
echo "=============================================="

source .venv/bin/activate

# Build command with common arguments
CMD="python -m src.meta.train_probe \
    --probe_type $PROBE_TYPE \
    --loss_type $LOSS_TYPE \
    --batch_size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --loss_file_pattern \"$LOSS_FILE_PATTERN\""


if [[ "$PROBE_TYPE" == "histogram" || "$PROBE_TYPE" == "delta" || "$PROBE_TYPE" == "kl_delta" ]]; then
    CMD="$CMD --num_bins $NUM_BINS --hidden_dims $HIDDEN_DIMS"
fi

if [[ "$PROBE_TYPE" == "delta" || "$PROBE_TYPE" == "kl_delta" ]]; then
    CMD="$CMD --min_gap $MIN_GAP"
    if [[ -n "$MAX_GAP" ]]; then
        CMD="$CMD --max_gap $MAX_GAP"
    fi
fi

# Add optional flags
if $USE_RMS_NORM && [[ "$PROBE_TYPE" == "cnn" || "$PROBE_TYPE" == "histogram" ]]; then
    CMD="$CMD --use_rms_norm"
fi

if [[ -n "$TARGET_LIST" ]]; then
    CMD="$CMD --target_list $TARGET_LIST"
fi

# Add any extra arguments
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Run training
echo "Running: $CMD"
eval $CMD

echo "Training completed!"
