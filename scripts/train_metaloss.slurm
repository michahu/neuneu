#!/bin/bash
#SBATCH --job-name=metaloss_train
#SBATCH --output=slurm/%A_%x.out
#SBATCH --error=slurm/%A_%x.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --account=torch_pr_287_general

# =============================================================================
# Train MetaLossPredictor - Unified Script
#
# Supports three loss types:
#   - ce:         Cross-Entropy Loss (classification)
#   - emd:        Earth Mover's Distance / Wasserstein (classification)
#   - ordinal_ce: Ordinal Cross-Entropy with soft Gaussian labels (classification)
#   - mse:        Mean Squared Error (regression)
#   - quantile:   Quantile regression (uncertainty-aware predictions)
#
# Supports baseline mode (no soft prompt encoder, just transformer encoder)
#
# Usage:
#   sbatch train_metaloss.slurm LOSS_TYPE [OPTIONS]
#
# Arguments:
#   LOSS_TYPE               mse or quantile
#
# Options:
#   --baseline              Use baseline model without soft prompt encoder
#   --predict_accuracy      Predict accuracy instead of loss
#   --input_type TYPE       Encoder input: loss, instance_accuracy, task_accuracy
#   --inverse_perplexity    Transform losses x -> e^(-x) before processing (values become 0-1 range)
#   --bin_min VALUE         Min value for histogram binning (default: 0.0 with inverse_perplexity)
#   --bin_max VALUE         Max value for histogram binning (default: 1.0 with inverse_perplexity)
#   --target_list TASKS     Tasks to predict (space-separated, quoted)
#   --heldout_list TASKS    Tasks to exclude (space-separated)
#   --data_dir DIR          Data directory (default: ./results/datadecide_train)
#   --output_dir DIR        Output directory (default: auto-generated)
#   --batch_size N          Batch size (default: 32)
#   --lr RATE               Learning rate (default: 6e-5)
#   --epochs N              Number of epochs (default: 5)
#   -- ARGS                 Pass all remaining args directly to train.py
#   (unknown flags)         Any unrecognized flags are passed to train.py
#
# Examples:
#   sbatch train_metaloss.slurm mse
#   sbatch train_metaloss.slurm quantile --predict_accuracy
#   sbatch train_metaloss.slurm ce --baseline
#   sbatch train_metaloss.slurm quantile --predict_accuracy --target_list "arc_challenge arc_easy boolq"
# =============================================================================

# Required arguments
LOSS_TYPE="${1:?Error: LOSS_TYPE must be provided as first argument (mse or quantile)}"
shift

# Validate loss type
if [[ ! "$LOSS_TYPE" =~ ^(ce|emd|ordinal_ce|mse|quantile)$ ]]; then
    echo "Error: Invalid LOSS_TYPE '$LOSS_TYPE'. Must be one of: ce, emd, ordinal_ce, mse, quantile"
    exit 1
fi

# Defaults
USE_BASELINE=""
PREDICT_ACCURACY=""
INPUT_TYPE="loss"  # loss, instance_accuracy, task_accuracy
TARGET_LIST=""  # If empty, will use DEFAULT_TARGET_LIST based on TASK_MODE
HELDOUT_LIST=""
OUTPUT_DIR=""
DATA_DIR="./results/datadecide_train"
BATCH_SIZE="64"
LEARNING_RATE="6e-4"
NUM_EPOCHS="3"
ENCODER_TYPE=""  # soft prompt encoder type (cnn, etc.)
INVERSE_PERPLEXITY=""
BIN_MIN=""
BIN_MAX=""
EXTRA_ARGS=""  # Catch-all for arbitrary train.py flags

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            USE_BASELINE="1"
            shift
            ;;
        --predict_accuracy)
            PREDICT_ACCURACY="1"
            shift
            ;;
        --input_type)
            INPUT_TYPE="$2"
            shift 2
            ;;
        --target_list)
            TARGET_LIST="$2"
            shift 2
            ;;
        --heldout_list)
            # Read all following arguments until next flag
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                HELDOUT_LIST="$HELDOUT_LIST $1"
                shift
            done
            ;;
        --encoder_type)
            ENCODER_TYPE="$2"
            shift 2
            ;;
        --inverse_perplexity)
            INVERSE_PERPLEXITY="1"
            shift
            ;;
        --bin_min)
            BIN_MIN="$2"
            shift 2
            ;;
        --bin_max)
            BIN_MAX="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --)
            # Everything after -- is passed directly to train.py
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        *)
            # Unknown options are passed through to train.py
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done



# Build short names for output directory and wandb
BASELINE_SUFFIX=$([[ -n "$USE_BASELINE" ]] && echo "_baseline" || echo "")

# Encoder type suffix (only if non-default)
ENCODER_SUFFIX=""
if [[ -n "$ENCODER_TYPE" ]]; then
    ENCODER_SUFFIX="_${ENCODER_TYPE}"
fi

# Signal propagation suffixes
INVP_SUFFIX=$([[ -n "$INVERSE_PERPLEXITY" ]] && echo "_invp" || echo "")

# Determine predict type based on input_type
case "$INPUT_TYPE" in
    task_accuracy)
        PREDICT_TYPE="task"
        ;;
    instance_accuracy)
        PREDICT_TYPE="inst"
        ;;
    *)
        if [[ -n "$PREDICT_ACCURACY" ]]; then
            PREDICT_TYPE="acc"
        else
            PREDICT_TYPE="loss"
        fi
        ;;
esac

# Build output directory name from settings
# Format: meta{loss|acc|inst|task}_{encoder}_{loss}[_baseline][_res]
RUN_NAME="meta${PREDICT_TYPE}${ENCODER_SUFFIX}_${LOSS_TYPE}${BASELINE_SUFFIX}${INVP_SUFFIX}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./results/${RUN_NAME}_lr${LEARNING_RATE}_$(date +%Y%m%d-%H%M%S)"
fi

echo "=============================================="
echo "Training Meta Predictor"
echo "=============================================="
echo "RUN_NAME: $RUN_NAME"
echo "INPUT_TYPE: $INPUT_TYPE"
echo "LOSS_TYPE: $LOSS_TYPE"
echo "ENCODER_TYPE: $([[ -n "$ENCODER_TYPE" ]] && echo "$ENCODER_TYPE" || echo "cnn (default)")"
echo "USE_BASELINE: $([[ -n "$USE_BASELINE" ]] && echo "yes" || echo "no")"
if [[ -n "$INVERSE_PERPLEXITY" ]]; then
    echo "INVERSE_PERPLEXITY: yes (bin_min=${BIN_MIN:-0.0}, bin_max=${BIN_MAX:-1.0})"
fi
if [[ -n "$PREDICT_ACCURACY" || "$INPUT_TYPE" != "loss" ]]; then
    if [[ -n "$TARGET_LIST" ]]; then
        echo "TARGET_LIST: $TARGET_LIST"
    fi
    if [[ -n "$HELDOUT_LIST" ]]; then
        echo "HELDOUT_LIST: $HELDOUT_LIST"
    fi
fi
echo "DATA_DIR: $DATA_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "NUM_EPOCHS: $NUM_EPOCHS"
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "EXTRA_ARGS: $EXTRA_ARGS"
fi
echo "=============================================="


source .venv/bin/activate

# Build base command
CMD="python -m src.meta.train \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --loss_type $LOSS_TYPE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --use_wandb \
    --wandb_project metaloss-predictor \
    --wandb_run_name ${RUN_NAME}_lr${LEARNING_RATE}_$(date +%Y%m%d-%H%M%S) \
    --fp16"

if [[ -n "$USE_BASELINE" ]]; then
    CMD="$CMD --use_baseline"
fi

# Add encoder type
if [[ -n "$ENCODER_TYPE" ]]; then
    CMD="$CMD --encoder_type $ENCODER_TYPE"
fi

# Add accuracy prediction flags
if [[ -n "$PREDICT_ACCURACY" ]]; then
    CMD="$CMD --predict_accuracy"
fi

# Add input_type flag (for diagnostic modes)
if [[ "$INPUT_TYPE" != "loss" ]]; then
    CMD="$CMD --input_type $INPUT_TYPE"
fi

# Add target_list and heldout_list for accuracy-based modes
if [[ -n "$PREDICT_ACCURACY" || "$INPUT_TYPE" != "loss" ]]; then
    if [[ -n "$TARGET_LIST" ]]; then
        CMD="$CMD --target_list $TARGET_LIST"
    fi
    if [[ -n "$HELDOUT_LIST" ]]; then
        CMD="$CMD --heldout_list $HELDOUT_LIST"
    fi
fi

# Add quantile-specific arguments
if [[ "$LOSS_TYPE" == "quantile" ]]; then
    CMD="$CMD --quantiles 0.1 0.25 0.5 0.75 0.9"
fi

# Add inverse perplexity transform flags
if [[ -n "$INVERSE_PERPLEXITY" ]]; then
    CMD="$CMD --inverse_perplexity"
    # Default bin range for inverse perplexity is [0, 1]
    CMD="$CMD --bin_min ${BIN_MIN:-0.0} --bin_max ${BIN_MAX:-1.0}"
elif [[ -n "$BIN_MIN" || -n "$BIN_MAX" ]]; then
    # Allow explicit bin range without inverse_perplexity
    [[ -n "$BIN_MIN" ]] && CMD="$CMD --bin_min $BIN_MIN"
    [[ -n "$BIN_MAX" ]] && CMD="$CMD --bin_max $BIN_MAX"
fi

# Add any extra arguments passed through
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Run training
echo "Running: $CMD"
eval $CMD

echo "Training completed!"

