#!/bin/bash
#SBATCH --job-name=hf_eval
#SBATCH --output=slurm/%A_%a_%x.out
#SBATCH --error=slurm/%A_%a_%x.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_287_general
#SBATCH --array=0-11

# =============================================================================
# Generic SLURM script for evaluating HuggingFace models (Pythia, OLMo, etc.)
# on the same validation data used for DataDecide models.
#
# Usage (array mode - evaluates multiple checkpoints):
#   # Evaluate checkpoints from step0 to step140000, every 5000 steps
#   # Script auto-computes: 29 checkpoints spread across array jobs
#   sbatch scripts/eval_hf_model.slurm EleutherAI/pythia-70m 0 140000 5000
#
# Usage (single mode - evaluates one checkpoint):
#   # Evaluate main/default checkpoint
#   sbatch scripts/eval_hf_model.slurm EleutherAI/pythia-70m
#
#   # Evaluate specific checkpoint/revision
#   sbatch scripts/eval_hf_model.slurm EleutherAI/pythia-70m step1000
#
# Array mode arguments: model start_step end_step step_gap [-- optional args]
# Single mode arguments: model [checkpoint] [-- optional args]
#
# Optional args (after --):
#   --data_dir DIR       Data directory (default: ./data/datadecide_subset)
#   --results_dir DIR    Results directory (default: ./results)
#   --shard SHARD        Shard number (default: 00000141)
#   --batch_size N       Batch size (default: 16)
#   --seq_len N          Sequence length (default: 1024)
#
# The script will:
#   1. Decode stored OLMo tokens back to text
#   2. Re-tokenize with the target model's tokenizer
#   3. Run evaluation and save per-token losses
#   4. Output to: results/hf_eval/{model_name}/{revision}/eval_losses_shard_*.npy
# =============================================================================

# Required argument: HuggingFace model name
HF_MODEL="${1:?Error: HF_MODEL must be provided as first argument (e.g., EleutherAI/pythia-70m)}"

# Defaults
DATA_DIR="./data/datadecide_subset"
RESULTS_DIR="./results"
SHARD_NUMBER="00000141"
BATCH_SIZE="16"
SEQ_LEN="1024"

# Detect mode: array mode if $2, $3, $4 are all numeric (start, end, gap)
if [[ "$2" =~ ^[0-9]+$ ]] && [[ "$3" =~ ^[0-9]+$ ]] && [[ "$4" =~ ^[0-9]+$ ]]; then
    # Array mode: arguments are start_step, end_step, step_gap
    START_STEP="$2"
    END_STEP="$3"
    STEP_GAP="$4"
    shift 4  # Remove positional args, leaving optional args

    MODE="array"
else
    # Single mode: original behavior
    CHECKPOINT="${2:-}"  # e.g., "step1000" or "main" (empty = default/main branch)
    shift 2 2>/dev/null || shift 1  # Remove positional args

    MODE="single"
fi

# Parse optional arguments after --
while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --shard)
            SHARD_NUMBER="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq_len)
            SEQ_LEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Array mode: build checkpoint list
if [ "$MODE" = "array" ]; then
    checkpoints=()
    for ((step=START_STEP; step<=END_STEP; step+=STEP_GAP)); do
        checkpoints+=("step${step}")
    done
    NUM_CHECKPOINTS=${#checkpoints[@]}

    # Get assignment for this SLURM_ARRAY_TASK_ID
    task_id=${SLURM_ARRAY_TASK_ID:-0}
    if (( task_id >= NUM_CHECKPOINTS )); then
        echo "Task $task_id out of range (0..$((NUM_CHECKPOINTS-1))), exiting"
        exit 0
    fi

    CHECKPOINT=${checkpoints[$task_id]}
fi

# Source tokenizer (the tokenizer used to create the stored tokens)
SOURCE_TOKENIZER="allenai/OLMo-1B"

echo "=============================================="
echo "HuggingFace Model Evaluation"
echo "=============================================="
echo "Model: $HF_MODEL"
echo "Mode: $MODE"
if [ "$MODE" = "array" ]; then
    echo "Array task ID: $task_id / $((NUM_CHECKPOINTS-1))"
    echo "Step range: $START_STEP to $END_STEP (gap: $STEP_GAP)"
    echo "Total checkpoints: $NUM_CHECKPOINTS"
fi
echo "Checkpoint: ${CHECKPOINT:-main}"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Shard: $SHARD_NUMBER"
echo "Batch size: $BATCH_SIZE"
echo "Seq len: $SEQ_LEN"
echo "Source tokenizer: $SOURCE_TOKENIZER"
echo "=============================================="


source .venv/bin/activate

# Build command with optional revision
CMD="python -m src.eval \
    --hf_model $HF_MODEL \
    --source_tokenizer $SOURCE_TOKENIZER \
    --data_dir $DATA_DIR \
    --eval_base_results_dir $RESULTS_DIR \
    --shard_number $SHARD_NUMBER \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --logging_steps 10"

# Add revision if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --revision $CHECKPOINT"
fi

eval $CMD

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $RESULTS_DIR/hf_eval/"
echo "=============================================="
