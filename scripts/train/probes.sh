#!/bin/bash
# =============================================================================
# probes.sh - Launch all probe experiments
#
# All probes use a linear output head (one prediction per downstream task).
#
# Compares:
#   1. DirectProbe (CNN-based, uses CNNSoftPromptGenerator)
#   2. Histogram delta probe (per-task by design)
#   3. Average Loss MLP probe (neural logistic equivalent)
#
# Usage:
#   ./scripts/train/probes.sh [--submit]
#
# =============================================================================

set -e

# Parse arguments
SUBMIT_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --submit)
            SUBMIT_FLAG="--submit"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# Settings
DATA_DIR="./results/datadecide_train"
OUTPUT_BASE="./results/final/probes"
SEEDS="0 1 2 3 4"
EPOCHS="5"
BATCH_SIZE="128"
LR="1e-3"
NUM_BINS="64"
HIDDEN_DIMS="128,128"

# Inverse perplexity settings (transform losses x -> e^(-x), values become 0-1)
# Set to "true" to enable, empty string to disable
INVERSE_PERP="true"  # or "true"
INVP_BIN_MIN="0.0"
INVP_BIN_MAX="1.0"

echo "=============================================="
echo "Probe Comparison Experiments"
echo "=============================================="
echo "Data dir: $DATA_DIR"
echo "Output base: $OUTPUT_BASE"
echo "Seeds: $SEEDS"
if [[ -n "$INVERSE_PERP" ]]; then
    echo "Inverse perplexity: enabled (bin_min=$INVP_BIN_MIN, bin_max=$INVP_BIN_MAX)"
fi
echo ""

# Build inverse perplexity args
INVP_ARGS=""
INVP_SUFFIX=""
if [[ -n "$INVERSE_PERP" ]]; then
    INVP_ARGS="--inverse_perplexity --bin_min $INVP_BIN_MIN --bin_max $INVP_BIN_MAX"
    INVP_SUFFIX="_invp"
fi

# Function to run or print command
run_cmd() {
    local desc="$1"
    shift
    echo "--- $desc ---"
    echo "CMD: $@"
    if [[ -n "$SUBMIT_FLAG" ]]; then
        eval "$@"
        echo ""
    else
        echo "(dry run - use --submit to actually run)"
        echo ""
    fi
}

# Loop over seeds
for SEED in $SEEDS; do
    echo ""
    echo "=============================================="
    echo "Seed: $SEED"
    echo "=============================================="

    # DirectProbe (CNN-based)
    echo ""
    echo ">>> Launching DirectProbe (CNN)..."
    run_cmd "DirectProbe (seed=$SEED)" \
        "sbatch scripts/train_probe.slurm \
            --probe_type cnn \
            --loss_type quantile \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LR \
            --seed $SEED \
            --data_dir $DATA_DIR \
            $INVP_ARGS \
            --output_dir $OUTPUT_BASE/cnn${INVP_SUFFIX}/seed$SEED"

    # # Histogram delta probe (per-task, RMSNorm built-in)
    # echo ""
    # echo ">>> Launching Histogram delta probe..."
    # run_cmd "Delta Probe (seed=$SEED)" \
    #     "sbatch scripts/train_probe.slurm \
    #         --probe_type delta \
    #         --num_bins $NUM_BINS \
    #         --hidden_dims $HIDDEN_DIMS \
    #         --loss_type quantile \
    #         --batch_size $BATCH_SIZE \
    #         --epochs $EPOCHS \
    #         --lr $LR \
    #         --seed $SEED \
    #         --data_dir $DATA_DIR \
    #         $INVP_ARGS \
    #         --output_dir $OUTPUT_BASE/delta${INVP_SUFFIX}/seed$SEED"

    # # Average Loss MLP probe (neural logistic equivalent)
    # echo ""
    # echo ">>> Launching Average Loss MLP probe..."
    # run_cmd "Avg Loss Probe (seed=$SEED)" \
    #     "sbatch scripts/train_probe.slurm \
    #         --probe_type avg_loss \
    #         --hidden_dims $HIDDEN_DIMS \
    #         --loss_type quantile \
    #         --batch_size $BATCH_SIZE \
    #         --epochs $EPOCHS \
    #         --lr $LR \
    #         --seed $SEED \
    #         --data_dir $DATA_DIR \
    #         $INVP_ARGS \
    #         --output_dir $OUTPUT_BASE/avg_loss${INVP_SUFFIX}/seed$SEED"
done

echo ""
echo "=============================================="
echo "All experiments launched!"
echo "=============================================="
