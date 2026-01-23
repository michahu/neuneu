#!/bin/bash
# =============================================================================
# probes.sh - Evaluate probe types (CNN, histogram, delta, etc.)
#
# Simple sequential submission - comment out any lines you don't need to run.
#
# Usage: bash scripts/eval/probes.sh
# =============================================================================


# Configuration
SEEDS="0 1 2 3 4"
CHECKPOINT_DIR="./results/final/probes"
OUTPUT_DIR="./results/evals/probes"
ANCHOR_STEP_IDX="0"
DELTA_EVAL_MODE="anchor"  # "anchor" or "cumulative"

# Common sbatch options
SBATCH_OPTS="--job-name=eval_probes --ntasks=1 --nodes=1 --cpus-per-task=4 --mem=32GB --time=2:00:00 --gres=gpu:1 --account=torch_pr_287_general"

echo "Submitting probe eval jobs"
echo "Seeds: $SEEDS"
echo ""

for SEED in $SEEDS; do
    echo "=== Seed $SEED ==="

    # =============================================================================
    # DataDecide models
    # =============================================================================
    DATA_DIR="./results/datadecide_dataset_test"
    MODELS="DataDecide-c4-90M DataDecide-c4-150M DataDecide-c4-300M DataDecide-c4-530M DataDecide-c4-750M DataDecide-c4-1B"

    # # --- avg_loss probe ---
    # CHECKPOINT="${CHECKPOINT_DIR}/avg_loss_invp/seed${SEED}/best_model.pt"
    # PROBE_OUTPUT_DIR="${OUTPUT_DIR}/avg_loss/seed${SEED}"
    # for MODEL in $MODELS; do
    #     sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR probe --checkpoint $CHECKPOINT --model_name $MODEL"
    # done

    # # --- delta probe ---
    # CHECKPOINT="${CHECKPOINT_DIR}/delta_invp/seed${SEED}/best_model.pt"
    # PROBE_OUTPUT_DIR="${OUTPUT_DIR}/delta_${DELTA_EVAL_MODE}/seed${SEED}"
    # for MODEL in $MODELS; do
    #     sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR delta_probe --checkpoint $CHECKPOINT --anchor_step_idx $ANCHOR_STEP_IDX --min_gap 1 --delta_eval_mode $DELTA_EVAL_MODE --model_name $MODEL"
    # done

    # --- cnn probe ---
    CHECKPOINT="${CHECKPOINT_DIR}/cnn_invp/seed${SEED}/best_model.pt"
    PROBE_OUTPUT_DIR="${OUTPUT_DIR}/cnn/seed${SEED}"
    for MODEL in $MODELS; do
        sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR probe --checkpoint $CHECKPOINT --model_name $MODEL"
    done

    # =============================================================================
    # DataDecide models
    # =============================================================================
    DATA_DIR="./results/datadecide_val"
    MODELS="DataDecide-dclm-baseline-90M DataDecide-dclm-baseline-150M DataDecide-dclm-baseline-300M DataDecide-dclm-baseline-530M DataDecide-dclm-baseline-750M"

    # # --- avg_loss probe ---
    # CHECKPOINT="${CHECKPOINT_DIR}/avg_loss_invp/seed${SEED}/best_model.pt"
    # PROBE_OUTPUT_DIR="${OUTPUT_DIR}/avg_loss/seed${SEED}"
    # for MODEL in $MODELS; do
    #     sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR probe --checkpoint $CHECKPOINT --model_name $MODEL"
    # done

    # # --- delta probe ---
    # CHECKPOINT="${CHECKPOINT_DIR}/delta_invp/seed${SEED}/best_model.pt"
    # PROBE_OUTPUT_DIR="${OUTPUT_DIR}/delta_${DELTA_EVAL_MODE}/seed${SEED}"
    # for MODEL in $MODELS; do
    #     sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR delta_probe --checkpoint $CHECKPOINT --anchor_step_idx $ANCHOR_STEP_IDX --min_gap 1 --delta_eval_mode $DELTA_EVAL_MODE --model_name $MODEL"
    # done

    # --- cnn probe ---
    CHECKPOINT="${CHECKPOINT_DIR}/cnn_invp/seed${SEED}/best_model.pt"
    PROBE_OUTPUT_DIR="${OUTPUT_DIR}/cnn/seed${SEED}"
    for MODEL in $MODELS; do
        sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR probe --checkpoint $CHECKPOINT --model_name $MODEL"
    done

    # =============================================================================
    # Pythia models
    # =============================================================================
    DATA_DIR="./results/hf_eval"
    MODELS="EleutherAI--pythia-70m EleutherAI--pythia-1.4b EleutherAI--pythia-2.8b EleutherAI--pythia-6.9b"

    # # --- avg_loss probe ---
    # CHECKPOINT="${CHECKPOINT_DIR}/avg_loss_invp/seed${SEED}/best_model.pt"
    # PROBE_OUTPUT_DIR="${OUTPUT_DIR}/avg_loss/seed${SEED}"
    # for MODEL in $MODELS; do
    #     sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR probe --checkpoint $CHECKPOINT --model_name $MODEL"
    # done

    # # --- delta probe ---
    # CHECKPOINT="${CHECKPOINT_DIR}/delta_invp/seed${SEED}/best_model.pt"
    # PROBE_OUTPUT_DIR="${OUTPUT_DIR}/delta_${DELTA_EVAL_MODE}/seed${SEED}"
    # for MODEL in $MODELS; do
    #     sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR delta_probe --checkpoint $CHECKPOINT --anchor_step_idx $ANCHOR_STEP_IDX --min_gap 1 --delta_eval_mode $DELTA_EVAL_MODE --model_name $MODEL"
    # done

    # --- cnn probe ---
    CHECKPOINT="${CHECKPOINT_DIR}/cnn_invp/seed${SEED}/best_model.pt"
    PROBE_OUTPUT_DIR="${OUTPUT_DIR}/cnn/seed${SEED}"
    for MODEL in $MODELS; do
        sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions --data_dir $DATA_DIR --output_dir $PROBE_OUTPUT_DIR probe --checkpoint $CHECKPOINT --model_name $MODEL"
    done


    echo ""
done

echo "Jobs submitted! Check with: squeue -u \$USER"
