#!/bin/bash
# =============================================================================
# baseline.sh - Evaluate transformer encoder only (no CNN)
#
# Simple sequential submission - comment out any lines you don't need to run.
#
# Usage: bash scripts/eval/baseline.sh
# =============================================================================


# Configuration
SEEDS="0 1 2 3 4"
CONTEXT_RATIO="0.2"
CHECKPOINT_DIR="./results/task_ablation_final"
CHECKPOINT_NAME="best_model.pt"
OUTPUT_DIR="./results/evals/task-ablation/"

# Common sbatch options
SBATCH_OPTS="--job-name=eval_baseline --ntasks=1 --nodes=1 --cpus-per-task=4 --mem=32GB --time=1:00:00 --gres=gpu:1 --account=torch_pr_287_general"

echo "Submitting baseline eval jobs"
echo "Seeds: $SEEDS"
echo ""

for SEED in $SEEDS; do
    CHECKPOINT="${CHECKPOINT_DIR}/baseline/seed${SEED}/${CHECKPOINT_NAME}"
    SEED_OUTPUT_DIR="${OUTPUT_DIR}/baseline/seed${SEED}"

    echo "=== Seed $SEED ==="
    echo "Checkpoint: $CHECKPOINT"
    echo ""

    # =============================================================================
    # DataDecide models
    # =============================================================================
    DATA_DIR="./results/datadecide_dataset_test"

    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-c4-90M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-c4-150M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-c4-300M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-c4-530M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-c4-750M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-c4-1B --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"

    # =============================================================================
    # DataDecide models, different seed
    # =============================================================================
    DATA_DIR="./results/datadecide_val"

    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-dclm-baseline-90M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-dclm-baseline-150M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-dclm-baseline-300M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-dclm-baseline-530M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name DataDecide-dclm-baseline-750M --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"

    # =============================================================================
    # Pythia models
    # =============================================================================
    DATA_DIR="./results/hf_eval"

    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name EleutherAI--pythia-70m --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name EleutherAI--pythia-1.4b --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name EleutherAI--pythia-2.8b --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
    sbatch $SBATCH_OPTS --wrap="source .venv/bin/activate && python -m src.analysis.eval_scaling_predictions baseline --model_name EleutherAI--pythia-6.9b --prediction_mode anchored --checkpoint $CHECKPOINT --data_dir $DATA_DIR --output_dir $SEED_OUTPUT_DIR --context_ratio $CONTEXT_RATIO"
done

echo ""
echo "Jobs submitted! Check with: squeue -u \$USER"
