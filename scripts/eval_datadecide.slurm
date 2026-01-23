#!/bin/bash
#SBATCH --job-name=perplexity_eval  # Job name
#SBATCH --output=slurm/%A_%a_%x.out
#SBATCH --error=slurm/%A_%a_%x.err
#SBATCH --ntasks=1  # Number of tasks (usually 1 for Python jobs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=64GB  # Memory per task (adjust as needed)
#SBATCH --time=48:00:00  # Maximum runtime (HH:MM:SS)
#SBATCH --gres=gpu:1  
#SBATCH --array=0-11
#SBATCH --account=torch_pr_287_general

# Set the output directory for the experiment
MODEL_NAME="datadecide_subset"
PRETRAINING_RECIPE="${1:?Error: PRETRAINING_RECIPE must be provided as first argument}"

declare -A ckpt_counts=(
  [90M]=25
  [150M]=32
  [300M]=38
  [530M]=48
  [750M]=52
  [1B]=29
)

BLOCK_SIZE=5
SEED_NAME="2"
model_sizes=(1B)
# model_sizes=(1B)

# build mapping of array indices → (size, start_idx)
map=()
for size in "${model_sizes[@]}"; do
  n=${ckpt_counts[$size]}
  blocks=$(( (n + BLOCK_SIZE - 1) / BLOCK_SIZE ))
  for ((b=0; b<blocks; b++)); do
    start=$((b * BLOCK_SIZE))
    map+=("$size:$start")
  done
done

# total jobs
total=${#map[@]}

# get assignment for this SLURM_ARRAY_TASK_ID
task_id=${SLURM_ARRAY_TASK_ID}
if (( task_id >= total )); then
  echo "Task id $task_id out of range (0..$((total-1)))"
  exit 1
fi

assignment=${map[$task_id]}
MODEL_SIZE=${assignment%%:*}
START_IDX=${assignment##*:}

if [[ "$MODEL_SIZE" == "90M" || "$MODEL_SIZE" == "150M" ]]; then
  BATCH_SIZE=16
elif [[ "$MODEL_SIZE" == "300M" || "$MODEL_SIZE" == "530M" ]]; then
  BATCH_SIZE=8
# elif [[ "$MODEL_SIZE" == "750M" ]]; then
#   BATCH_SIZE=4
else
  BATCH_SIZE=4
fi

echo "MODEL_SIZE: $MODEL_SIZE"
echo "START_IDX: $START_IDX"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "--------------------------------"



source .venv/bin/activate

python -m src.eval \
    --model_name $MODEL_NAME \
    --seed_name $SEED_NAME \
    --data_dir ./data/$MODEL_NAME/ \
    --eval_base_results_dir ./results/datadecide_val \
    --shard_number 00000141 \
    --datadecide_pretraining_recipe $PRETRAINING_RECIPE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --logging_steps 10 \
    --model_size $MODEL_SIZE \
    --trained_model_start_idx $START_IDX \
    --num_models_to_eval $BLOCK_SIZE 