#!/bin/bash
# =============================================================================
# full.sh - Evaluate full meta-model with different soft prompt types
#
# Simple sequential submission - comment out any lines you don't need to run.
#
# Usage: bash scripts/eval/full.sh [soft_prompt_type]
#   soft_prompt_type: cnn, avg_loss, or delta (default)
# =============================================================================


# Configuration
SOFT_PROMPT_TYPE="${1:-delta}"  # Options: cnn, avg_loss, delta
SEEDS="0"
CONTEXT_RATIO="0.2"
# CHECKPOINT_DIR="./results/ablation/"
CHECKPOINT_DIR="./results/ablation/"
CHECKPOINT_NAME="best_model.pt"

# Common sbatch options
SBATCH_OPTS="--job-name=eval_metamodel --ntasks=1 --nodes=1 --cpus-per-task=4 --mem=32GB --time=1:00:00 --gres=gpu:1 --account=torch_pr_287_general"

echo "Submitting eval jobs for ${SOFT_PROMPT_TYPE} soft prompts"
echo "Seeds: $SEEDS"
echo ""

for SEED in $SEEDS; do
    CHECKPOINT="${CHECKPOINT_DIR}/metamodel/${SOFT_PROMPT_TYPE}/seed${SEED}/${CHECKPOINT_NAME}"
    OUTPUT_DIR="./results/evals/invp_ablation/neural/${SOFT_PROMPT_TYPE}/seed${SEED}"

    echo "=== Seed $SEED ==="
    echo "Checkpoint: $CHECKPOINT"
    echo ""

    # =============================================================================
    # DataDecide models
    # =============================================================================
    DATA_DIR="./results/datadecide_dataset_test"

    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-c4-90M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-c4-150M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-c4-300M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-c4-530M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-c4-750M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-c4-1B --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"

    # =============================================================================
    # DataDecide models, different seed
    # =============================================================================
    DATA_DIR="./results/datadecide_val"

    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-dclm-baseline-90M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-dclm-baseline-150M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-dclm-baseline-300M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-dclm-baseline-530M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name DataDecide-dclm-baseline-750M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"

    # =============================================================================
    # Pythia models (uncomment to run)
    # =============================================================================
    DATA_DIR="./results/hf_eval"

    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name EleutherAI--pythia-70m --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name EleutherAI--pythia-1.4b --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name EleutherAI--pythia-2.8b --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions neural --model_name EleutherAI--pythia-6.9b --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
done

echo ""
echo "Jobs submitted! Check with: squeue -u \$USER"
