#!/bin/bash
#SBATCH --job-name=eval_logistic
#SBATCH --output=slurm/%A_%a_%x.out
#SBATCH --error=slurm/%A_%a_%x.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --account=torch_pr_287_general

# =============================================================================
# logistic.sh - Evaluate zero-shot logistic baseline
#
# Fits 4-parameter logistic on train data and evaluates on eval data.
# No seeds needed since logistic fitting is deterministic.
#
# Array size depends on model family:
#   - datadecide: 0-5 (6 models)
#   - pythia: 0-3 (4 models)
#
# Usage:
#   sbatch --array=0-5 scripts/eval/logistic.sh --model_family datadecide [OPTIONS]
#   sbatch --array=0-3 scripts/eval/logistic.sh --model_family pythia [OPTIONS]
#
# Options:
#   --model_family FAMILY   Model family: "datadecide" or "pythia" (required)
#   --train_data_dir DIR    Training data directory (default: ./results/datadecide_train)
#   --data_dir DIR          Directory containing evaluation data (default: auto-set by model_family)
#   --output_dir DIR        Directory for output results (default: ./results/dataset_test)
#
# Examples:
#   sbatch --array=0-5 scripts/eval/logistic.sh --model_family datadecide
#   sbatch --array=0-3 scripts/eval/logistic.sh --model_family pythia
#   sbatch --array=0 scripts/eval/logistic.sh --model_family datadecide  # Only 90M model
# =============================================================================

# Defaults
MODEL_FAMILY=""
TRAIN_DATA_DIR="./results/datadecide_train"
OUTPUT_DIR="./results/dataset_test"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_family)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        --train_data_dir)
            TRAIN_DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

source .venv/bin/activate

# Define models and default data_dir based on family
if [[ "$MODEL_FAMILY" == "datadecide" ]]; then
    MODELS=(DataDecide-c4-90M DataDecide-c4-150M DataDecide-c4-300M DataDecide-c4-530M DataDecide-c4-750M DataDecide-c4-1B)
    [[ -z "$DATA_DIR" ]] && DATA_DIR="./results/datadecide_dataset_test"
elif [[ "$MODEL_FAMILY" == "datadecide-d" ]]; then
    MODELS=(DataDecide-dclm-baseline-90M DataDecide-dclm-baseline-150M DataDecide-dclm-baseline-300M DataDecide-dclm-baseline-530M DataDecide-dclm-baseline-750M)
    [[ -z "$DATA_DIR" ]] && DATA_DIR="./results/datadecide_val"
elif [[ "$MODEL_FAMILY" == "pythia" ]]; then
    MODELS=(EleutherAI--pythia-70m EleutherAI--pythia-1.4b EleutherAI--pythia-2.8b EleutherAI--pythia-6.9b)
    [[ -z "$DATA_DIR" ]] && DATA_DIR="./results/hf_eval"
else
    echo "Error: --model_family must be 'datadecide' or 'pythia', got '$MODEL_FAMILY'"
    exit 1
fi

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Zero-Shot Logistic Baseline Evaluation"
echo "=============================================="
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model family: $MODEL_FAMILY"
echo "Model: $MODEL"
echo "Training data: $TRAIN_DATA_DIR"
echo "Evaluation data: $DATA_DIR"
echo "Output dir: ${OUTPUT_DIR}/logistic"
echo "=============================================="

python -m src.analysis.eval_scaling_predictions logistic \
    --model_name "$MODEL" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Evaluation complete!"
